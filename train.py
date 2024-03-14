"""
Run GFN training on sequences generation problem

Usage:
    train.py <cfg_path> <sequences_path> <output_path> [--nb_device=<device_num>] [--quiet] [--amp]
    train.py resume <resume_path> <sequences_path> <output_path> [--nb_device=<device_num>] [--quiet] [--amp] [--cfg_file=<cfg_file>]

Options:
    <cfg_path>                              config path
    <sequences_path>                        sequence file path
    <output_path>                           output folder
    <resume_path>                            directory of an earlier experiment to resume
    --nb_device=<device_num>                specify the number of cuda devices available for training [default: 1]
    --quiet                                 do not show progress information during training or evaluation
    --amp                                   use amp fp16 training
    --cfg_file=<cfg_file>                   use specific cfg file
    -h --help                               Show this screen
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import datetime
import shutil
import gzip
from src.utils.utils import schedule, cascading_schedule
import src.utils.plot_utils as plot_utils
from src.utils.ddp_setup import setup, cleanup, run_mp
from src.gfn.gfn_evaluator import GFNEvaluator
from src.configs.defaults import get_cfg_defaults
from src.utils.utils import correct_cfg_data, load_sequences
from src.env import build_env
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader
from src.gfn.build import build_gfn
from src.utils.logging import get_logger
import torch.distributed as dist
from heapq import heappush, heappushpop
from docopt import docopt


def generate_exploration_spec(exploration_cfg, epoch):
    assert exploration_cfg.METHOD in ['EPS_ANNEALING', 'TEMPERATURE_ANNEALING', 'NONE']
    if exploration_cfg.METHOD == 'NONE':
        return None
    start_value = exploration_cfg.START_VALUE
    end_value = exploration_cfg.END_VALUE
    anneal_type = exploration_cfg.ANNEAL_TYPE
    T = exploration_cfg.T
    start_value_epoch = schedule(start_value, end_value, T, epoch, type=anneal_type)
    end_value_epoch = schedule(start_value, end_value, T, epoch + 1, type=anneal_type)
    exploration_spec = {
        'exploration_method': exploration_cfg.METHOD,
        'start_value': start_value_epoch,
        'end_value': end_value_epoch
    }
    return exploration_spec


def train_epoch(cfg, epoch_id, generator, data_loader, logger, log_results, verbose):
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    exploration_cfg = training_cfg.EXPLORATION

    exploration_specs = generate_exploration_spec(exploration_cfg, epoch_id)
    epoch_data_iterator = data_loader.build_epoch_iterator(generator, exploration_specs)
    total_sequences_per_epoch = data_loader.steps_per_epoch * data_loader.batch_size
    mini_num_splits = training_cfg.MINI_BATCH_SPLITS
    batch_size = mini_num_splits * data_loader.batch_size

    mini_batch_counter = 0
    if verbose:
        bar = tqdm(total=total_sequences_per_epoch, leave=True, position=0, desc=f'Epoch: {epoch_id + 1}')
    else:
        bar = None

    # update LR in case using stepLR
    if generator.scheduler is not None:
        lr_cfg = cfg.GFN.MODEL.LR_SCHEDULER
        if lr_cfg.TYPE == 'STEP':
            generator.scheduler.step(epoch_id)

    for t, ((batch, trajs), random_spec) in enumerate(epoch_data_iterator):

        if not cfg.AMP:
            generator.accumulate_loss(batch, mini_num_splits)
        else:
            generator.accumulate_loss_amp(batch, mini_num_splits)
        mini_batch_counter += 1
        if mini_batch_counter % mini_num_splits == 0:
            if not cfg.AMP:
                ret = generator.update_model()
            else:
                ret = generator.update_model_amp()
            mini_batch_counter = 0
            # for all other LR updates types update here
            if generator.scheduler is not None:
                lr_cfg = cfg.GFN.MODEL.LR_SCHEDULER
                if lr_cfg.TYPE != 'STEP':
                    generator.scheduler.step(epoch_id + t / data_loader.steps_per_epoch)

            status_str = 'Epoch {}, '.format(epoch_id + 1)
            for key, value in ret.items():
                if key == 'loss':
                    status_str += f'{key}: {value:.4f}, '

                if log_results:
                    logger.add_scalar(key, value)
                    if key == 'loss':
                        plot_utils.plot('log_loss', np.log(value))
                    else:
                        plot_utils.plot(key, value)

            if log_results:
                for idx, p in enumerate(generator.opt.param_groups):
                    logger.add_scalar('lr_param_group_{}'.format(idx + 1), p['lr'])
                if random_spec is not None:
                    if 'random_action_prob' in random_spec:
                        logger.add_scalar('eps', random_spec['random_action_prob'])
                    if 'T' in random_spec:
                        logger.add_scalar('temperature', random_spec['T'])

            if log_results:
                with torch.no_grad():
                    log_z = generator.compute_log_Z().reshape(-1).cpu().numpy()
                    logger.add_scalar('log_partition', log_z[0])
                    plot_utils.plot('log_partition', log_z[0])

            if bar is not None:
                bar.update(batch_size)
                bar.set_description(status_str)

    if generator.loss != 0:
        if not cfg.AMP:
            ret = generator.update_model()
        else:
            ret = generator.update_model_amp()

    if log_results:
        plot_utils.tick(index=1)  # index=1 is for the epochs
        plot_utils.flush()

    if bar is not None:
        bar.close()


def aggregate_best_trees_devices(best_trees_device_path, best_trees_path, best_trees_size, world_size):
    """

    :param best_trees_device_path: stored best trees path of each device tower
    :param best_trees_path:  best trees path
    :param best_trees_size:  replay buffer size
    :param world_size:  total num of towers
    :return:
    """
    best_trees = []
    seen_trees_keys = {}
    for idx in range(world_size):
        best_seen_trees_tmp = pickle.load(open(best_trees_device_path.format(idx), 'rb'))
        for tree in best_seen_trees_tmp:
            signature = tree.signature
            if signature not in seen_trees_keys:
                seen_trees_keys[signature] = None
                if len(best_trees) >= best_trees_size:
                    dropped_tree = heappushpop(best_trees, tree)
                    key = dropped_tree.leafsets_signature
                    del seen_trees_keys[key]
                else:
                    heappush(best_trees, tree)

    pickle.dump(best_trees, open(best_trees_path, 'wb'))


def train(device_rank, world_size, ddp, cfg, paths_data, verbose=False):
    """

    :param device_rank: current running device
    :param world_size: total number of devices available for training
    :param ddp:
    :param cfg:
    :param paths_data:
    :param verbose:
    :return:
    """
    if ddp:
        setup(device_rank, world_size)
    #
    log_results = device_rank == 0
    verbose = verbose and device_rank == 0

    # build env
    all_seqs = load_sequences(paths_data['sequences_path'])
    env = build_env(cfg, all_seqs)
    env.to(device_rank)

    # build model
    generator = build_gfn(cfg, env, device_rank, ddp=ddp)

    # build rollout worker
    rollout_worker = RolloutWorker(env)

    # build logger
    logger = get_logger(cfg) if log_results else None

    # load weights if resume
    resume_path = paths_data['resume_path']
    eval_states = None
    if resume_path:
        checkpoints_folder = os.path.join(resume_path, 'checkpoints')
        checkpoints_paths = [os.path.join(checkpoints_folder, file) for file in os.listdir(checkpoints_folder) if
                             file.endswith('.pt')]
        checkpoints_paths = sorted(checkpoints_paths)
        latest_checkpoint_path = checkpoints_paths[-1]
        if verbose:
            print(f'loading checkpoint: {latest_checkpoint_path}')
        generator.load(latest_checkpoint_path)
        epoch_to_start = int(latest_checkpoint_path.split('_')[-1].split('.')[0]) + 1
        plot_utils.load(os.path.join(resume_path, 'plot_utils_save.pkl'))

        # load logger data
        logger.data = pickle.load(gzip.open(os.path.join(resume_path, 'logs')))
        # load evaluation states
        cfg_eval = cfg.GFN.MODEL.EVALUATION
        if cfg_eval.FIXED_STATES:
            all_files = os.listdir(resume_path)
            eval_folders = sorted([x for x in all_files if 'eval_scatter_' in x])
            eval_states = pickle.load(
                open(os.path.join(resume_path, eval_folders[-1], 'eval_states.pt'), 'rb'))

        # update environment temperature
        if epoch_to_start > 0:
            training_cfg = cfg.GFN.TRAINING_DATA_LOADER
            t_anneal_cfg = training_cfg.TEMPERATURE_ANNEALING
            if t_anneal_cfg.TEMPERATURE_ANNEALING:
                if t_anneal_cfg.ANNEAL_TYPE != 'CASCADING':
                    inverse_anneal = t_anneal_cfg.INVERSE_TEMPERATURE_ANNEALING
                    if inverse_anneal:
                        temperature = schedule(t_anneal_cfg.START_VALUE, t_anneal_cfg.END_VALUE, t_anneal_cfg.T,
                                               epoch_to_start,
                                               type=t_anneal_cfg.ANNEAL_TYPE)
                    else:
                        temperature = schedule(1/t_anneal_cfg.START_VALUE, 1/t_anneal_cfg.END_VALUE, t_anneal_cfg.T,
                                               epoch_to_start,
                                               type=t_anneal_cfg.ANNEAL_TYPE)
                        temperature = 1/temperature
                else:
                    temperature = cascading_schedule(t_anneal_cfg.CASCADING_SCHEDULE, epoch_to_start)
                env.reward_fn.scale = temperature
    else:
        epoch_to_start = 0
        # initialize plotting setups
        plot_utils._enlarge_ticker(1)
        plot_utils.set_xlabel_for_tick(1, 'epochs')

    plot_utils.set_output_dir(cfg.OUTPUT_PATH)
    plot_utils.suppress_stdout()

    # build training data loader
    best_trees_path = paths_data['best_trees_path']
    data_loader = TrainingDataLoader(cfg, env, rollout_worker, best_trees_path)

    # build evaluator
    if device_rank == 0:
        evaluation_cfg = cfg.GFN.MODEL.EVALUATION
        gfn_evaluator = GFNEvaluator(evaluation_cfg, rollout_worker, generator, states=eval_states, verbose=verbose)
    else:
        gfn_evaluator = None

    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    t_anneal_cfg = training_cfg.TEMPERATURE_ANNEALING
    epochs_num = training_cfg.EPOCHS_NUM
    evaluation_freq = cfg.GFN.MODEL.EVALUATION.EVALUATION_FREQ
    for epoch_id in range(epoch_to_start, epochs_num):
        if t_anneal_cfg.TEMPERATURE_ANNEALING:
            if t_anneal_cfg.ANNEAL_TYPE != 'CASCADING':
                inverse_anneal = t_anneal_cfg.INVERSE_TEMPERATURE_ANNEALING
                if inverse_anneal:
                    temperature = schedule(t_anneal_cfg.START_VALUE, t_anneal_cfg.END_VALUE, t_anneal_cfg.T,
                                           epoch_id,
                                           type=t_anneal_cfg.ANNEAL_TYPE)
                else:
                    temperature = schedule(1/t_anneal_cfg.START_VALUE, 1/t_anneal_cfg.END_VALUE, t_anneal_cfg.T,
                                           epoch_id,
                                           type=t_anneal_cfg.ANNEAL_TYPE)
                    temperature = 1/temperature
            else:
                temperature = cascading_schedule(t_anneal_cfg.CASCADING_SCHEDULE, epoch_id)
            if temperature != env.reward_fn.scale:
                if verbose:
                    print('Update temperature to ', temperature)
                current_log_z = generator.compute_log_Z().item()
                current_log_z = current_log_z * (env.reward_fn.scale / temperature)
                _ = torch.nn.init.constant_(generator._Z, current_log_z / 256)
                env.reward_fn.scale = temperature
            if device_rank == 0:
                logger.add_scalar('temperature', temperature)

        train_epoch(cfg, epoch_id, generator, data_loader, logger, log_results, verbose)
        save_path = os.path.join(paths_data['output_path'], 'checkpoints', "checkpoint_%06d.pt" % (epoch_id,))

        # save model and logging data
        if device_rank == 0:
            generator.save(save_path)
            logger.save()
            plot_utils.save()

            # run evaluation
            if epoch_id % evaluation_freq == 0:
                evaluation_result = gfn_evaluator.evaluate_gfn_quality(True)
                save_evaluation_results(logger, gfn_evaluator.states, plot_utils, evaluation_result,
                                        paths_data['output_path'], epoch_id)

                states = gfn_evaluator.states + evaluation_result['gfn_samples_result']['states']
                gfn_evaluator.update_states_set(states)
                print('Epoch {}, MLL {}, PEARSONR {}'.format(epoch_id, evaluation_result.get('mll'),
                                                             evaluation_result['log_pearsonr']))

        # save best trees
        if data_loader.best_state_batch_size > 0:
            best_trees_device_path = paths_data['best_trees_device_path'].format(device_rank)
            pickle.dump(data_loader.best_trees, open(best_trees_device_path, 'wb'))
            if ddp:
                dist.barrier()

            # aggregate all best trees across all devices
            if device_rank == 0:
                aggregate_best_trees_devices(best_trees_device_path, best_trees_path,
                                             training_cfg.BEST_TREES_BUFFER_SIZE,
                                             world_size)
            if ddp:
                dist.barrier()
            # update best trees in data loader
            best_trees = pickle.load(open(best_trees_path, 'rb'))
            data_loader.update_best_trees(best_trees)

    if device_rank == 0:
        final_mlls = []
        for _ in range(10):
            mll = gfn_evaluator.evaluate_marginal_likelihood(1024)
            final_mlls.append(mll)
        print('Final MLL evaluation: mean {}, std {}'.format(np.mean(final_mlls), np.std(final_mlls)))
        pickle.dump(final_mlls, open('final_mlls.p', 'wb'))

    if ddp:
        dist.barrier()
        cleanup()


def save_evaluation_results(logger, states, plot_utils, evaluation_result, output_path, epoch_id):
    logger.add_scalar('gfn_quality_log_pearsonr', evaluation_result['log_pearsonr'])
    if 'mll' in evaluation_result:
        logger.add_scalar('mll', evaluation_result['mll'])

    plot_utils.plot('gfn_quality_log_pearsonr', evaluation_result['log_pearsonr'], index=1)
    eval_scatter_path = os.path.join(output_path, f'eval_scatter_{epoch_id:06d}')

    # scatter plots for model probabilities and rewards
    if not os.path.exists(eval_scatter_path):
        os.makedirs(eval_scatter_path)

    path = os.path.join(eval_scatter_path, 'eval_log_prob_reward.png')
    plot_utils.plot_scatter(evaluation_result['log_prob_reward'][0], evaluation_result['log_prob_reward'][1],
                            'model logp', 'log reward', path
                            )

    path = os.path.join(eval_scatter_path, 'eval_prob_reward.png')
    plot_utils.plot_scatter(
        np.exp(evaluation_result['log_prob_reward'][0]), np.exp(evaluation_result['log_prob_reward'][1]),
        'model probability', 'reward', path
    )
    pickle.dump(evaluation_result['log_prob_reward'],
                open(os.path.join(eval_scatter_path, 'eval_log_prob_reward.pkl'), 'wb'))

    gfn_samples_result = evaluation_result['gfn_samples_result']
    if 'mut_mean' in gfn_samples_result:
        logger.add_scalar('gfn_sampled_mutations_mean', gfn_samples_result['mut_mean'])
        logger.add_scalar('gfn_sampled_mutations_std', gfn_samples_result['mut_std'])
        logger.add_scalar('gfn_sampled_mutations_min', gfn_samples_result['mut_min'])
        logger.add_scalar('gfn_sampled_mutations_max', gfn_samples_result['mut_max'])
        plot_utils.plot('gfn_sampled_mutations_mean', gfn_samples_result['mut_mean'], index=1)
        plot_utils.plot('gfn_sampled_mutations_mean_std', (gfn_samples_result['mut_mean'],
                                                           gfn_samples_result['mut_std']), index=1)
        plot_utils.plot('gfn_sampled_mutations_mean_min_max', (gfn_samples_result['mut_mean'],
                                                               gfn_samples_result['mut_min'],
                                                               gfn_samples_result['mut_max']), index=1)
        try:
            logger.draw_histogram('sampled states mutations', np.array(gfn_samples_result['mutations']), epoch_id)
        except:
            pass
    if 'log_scores' in gfn_samples_result:
        logger.add_scalar('gfn_sampled_log_scores_mean', gfn_samples_result['log_scores_mean'])
        logger.add_scalar('gfn_sampled_log_scores_std', gfn_samples_result['log_scores_std'])
        logger.add_scalar('gfn_sampled_log_scores_min', gfn_samples_result['log_scores_min'])
        logger.add_scalar('gfn_sampled_log_scores_max', gfn_samples_result['log_scores_max'])
        plot_utils.plot('gfn_sampled_log_scores_mean', gfn_samples_result['log_scores_mean'], index=1)
        plot_utils.plot('gfn_sampled_log_scores_mean_std', (gfn_samples_result['log_scores_mean'],
                                                            gfn_samples_result['log_scores_std']), index=1)
        plot_utils.plot('gfn_sampled_log_scores_mean_min_max', (gfn_samples_result['log_scores_mean'],
                                                                gfn_samples_result['log_scores_min'],
                                                                gfn_samples_result['log_scores_max']), index=1)
        try:
            logger.draw_histogram('sampled states log scores', np.array(gfn_samples_result['log_scores']), epoch_id)
        except:
            pass

    path = os.path.join(eval_scatter_path, 'eval_states.pt')
    pickle.dump(states, open(path, 'wb'))


if __name__ == '__main__':

    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    verbose = not arguments['--quiet']
    nb_device = int(arguments['--nb_device'])
    sequences_path = arguments['<sequences_path>']
    all_seqs = load_sequences(sequences_path)

    # load cfg
    if arguments['resume']:
        resume_path = arguments['<resume_path>']
        if arguments['--cfg_file'] is not None:
            cfg_path = arguments['--cfg_file']
        else:
            cfg_path = os.path.join(resume_path, 'config.yaml')
    else:
        resume_path = None
        cfg_path = arguments['<cfg_path>']
    output_path = arguments['<output_path>']
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.AMP = arguments['--amp']
    cfg = correct_cfg_data(all_seqs, nb_device, cfg)

    assert output_path != ''
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_ = output_path.split(os.path.sep)
    output_path_[-1] = cur_time + '_' + output_path_[-1]
    output_path = os.path.sep.join(output_path_)
    cfg.OUTPUT_PATH = output_path

    # create folders
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    # backup major files for future reference
    backup_dir = os.path.join(output_path, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # copy the full src directory
    shutil.copy(__file__, backup_dir)
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(os.path.join(current_dir_path, 'src'), os.path.join(backup_dir, 'src'))

    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))

    if resume_path:
        best_trees_path = os.path.join(output_path, 'best_trees.pt')
        best_trees_path_prev = os.path.join(resume_path, 'best_trees.pt')
        os.system('cp {} {}'.format(best_trees_path_prev, best_trees_path))

    paths_data = {
        'sequences_path': sequences_path,
        'output_path': output_path,
        'best_trees_path': os.path.join(output_path, 'best_trees.pt'),
        'best_trees_device_path': os.path.join(output_path, 'best_trees_{}.pt'),
        'resume_path': resume_path
    }

    if nb_device > 1:
        print('Train with DDP')
        run_mp(train, nb_device, args=(nb_device, True, cfg, paths_data, verbose))
    else:
        print('Train with 1 gpu')
        train(0, 1, False, cfg, paths_data, verbose)
