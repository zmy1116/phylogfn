import torch
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr


class GFNEvaluator(object):

    def __init__(self, evaluation_cfg, rollout_worker, generator, states=None, verbose=True):

        self.env = rollout_worker.env
        self.rollout_worker = rollout_worker
        self.generator = generator
        self.evaluation_cfg = evaluation_cfg
        self.verbose = verbose
        self.parsimony_problem = self.env.parsimony_problem
        self.perturb_branch_lengths_mll = self.env.cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION == 'CATEGORICAL'

        if states is not None:
            self.states = states
        else:
            self.states = self.generate_initial_states()

        self.batch_size = evaluation_cfg.BATCH_SIZE

    def generate_initial_states(self):

        trajs = self.env.sample(self.evaluation_cfg.STATES_NUM, True)
        return [x.current_state for x in trajs]

    def evaluate_marginal_likelihood(self, traj_size=1024):

        with torch.no_grad():
            log_pfs_all = []
            log_pbs_all = []
            log_scores_all = []
            discrete_factors_all = []
            for _ in range(0, traj_size, 256):
                data, trajectories = self.rollout_worker.rollout(self.generator, 256,
                                                                 generate_full_trajectories=True)
                log_paths_pf, log_paths_pb = data['log_paths_pf'], data['log_paths_pb']

                trees = [x.current_state.subtrees[0] for x in trajectories]
                perturbed_log_scores = []
                for tree in trees:
                    log_score, feature_dict, discrete_factor = self.env.compute_tree_log_score(tree.ete_node,
                                                                                               self.perturb_branch_lengths_mll)
                    perturbed_log_scores.append(log_score)
                    discrete_factors_all.append(discrete_factor)
                log_pf = log_paths_pf.sum(-1)
                log_pb = log_paths_pb.sum(-1)
                log_scores = torch.tensor(perturbed_log_scores).to(log_pf)
                log_pfs_all.append(log_pf)
                log_pbs_all.append(log_pb)
                log_scores_all.append(log_scores)

            log_pf = torch.cat(log_pfs_all)
            log_pb = torch.cat(log_pbs_all)
            log_scores = torch.cat(log_scores_all)
            num_trees = len(self.env.seq_arrays)
            discrete_factors_all = torch.tensor(discrete_factors_all).to(log_pf)
            tree_factor = - np.sum(np.log(np.arange(3, 2 * num_trees - 3, 2)))
            marginal_likelihood = torch.logsumexp(log_scores + log_pb + discrete_factors_all - log_pf, dim=0) - np.log(
                traj_size) + tree_factor
            return marginal_likelihood.item()

    def evaluate_gfn_quality_pearsonr(self, states=None):

        if states is None:
            states = self.states

        states_gfn_logp, states_log_rewards = [], []
        for state in tqdm(states):

            tree = state.subtrees[0]

            input_actions_set = []
            trajs_num = self.evaluation_cfg.TRAJECTORIES_PER_STATES
            for _ in range(trajs_num):
                actions_list, log_paths_pb = self.env.sample_backward_from_tree(tree)
                input_actions_set.append(actions_list)
            with torch.no_grad():
                data, _ = self.rollout_worker.rollout(self.generator, trajs_num,
                                                      generate_full_trajectories=False,
                                                      input_actions_set=input_actions_set)

                log_paths_pf, log_paths_pb = data['log_paths_pf'], data['log_paths_pb']

                log_pf = log_paths_pf.sum(-1)
                log_pb = log_paths_pb.sum(-1)
                log_rewards = data['log_rewards']
                state_gfn_logp = torch.logsumexp(log_pf - log_pb, dim=0) - np.log(trajs_num)
            states_gfn_logp.append(state_gfn_logp.item())
            states_log_rewards.append(log_rewards[0].item())

        return states_gfn_logp, states_log_rewards, pearsonr(states_gfn_logp, states_log_rewards)[0]

    def evaluate_gfn_quality(self, estimate_mll):

        # estimate gfn by pearson r
        states_gfn_logp, states_log_rewards, log_pearsonr = self.evaluate_gfn_quality_pearsonr()

        eval_ret = {
            'log_prob_reward': [states_gfn_logp, states_log_rewards],
            'log_pearsonr': pearsonr(states_gfn_logp, states_log_rewards)[0],
        }

        if estimate_mll:
            mll = self.evaluate_marginal_likelihood(1024)
            eval_ret['mll'] = mll

        with torch.no_grad():
            states = []
            for _ in range(0, self.evaluation_cfg.MUTATIONS_TRAJS, 256):
                data, trajs = self.rollout_worker.rollout(self.generator, 256,
                                                          generate_full_trajectories=True)
                states = states + [x.current_state for x in trajs]
            if self.parsimony_problem:
                mutations = [x.subtrees[0].total_mutations for x in states]
                mut_mean, mut_std = np.mean(mutations), np.std(mutations)
                mut_min, mut_max = np.min(mutations), np.max(mutations)
                sample_result = {
                    'states': states,
                    'mutations': mutations,
                    'mut_mean': mut_mean,
                    'mut_std': mut_std,
                    'mut_min': mut_min,
                    'mut_max': mut_max
                }
            else:
                log_scores = [x.subtrees[0].log_score for x in states]
                log_scores_mean = np.mean(log_scores)
                log_scores_std = np.std(log_scores)
                log_scores_min, log_scores_max = np.min(log_scores), np.max(log_scores)
                sample_result = {
                    'states': states,
                    'log_scores': log_scores,
                    'log_scores_mean': log_scores_mean,
                    'log_scores_std': log_scores_std,
                    'log_scores_min': log_scores_min,
                    'log_scores_max': log_scores_max
                }
            eval_ret['gfn_samples_result'] = sample_result
        return eval_ret

    def update_states_set(self, states):

        random.shuffle(states)
        self.states = states[:self.evaluation_cfg.STATES_NUM]
