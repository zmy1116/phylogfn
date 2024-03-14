import numpy as np
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from src.gfn.tb_gfn_phylo import TBGFlowNetGenerator


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def is_similar(seq_a, seq_b, dist_type="edit", threshold=0):
    if dist_type == "edit":
        return edit_dist(seq_a, seq_b) < threshold
    return False


def permu(lists):
    def fn(lists, group=[], result=[]):
        if not lists:
            result.append(group)
            return
        first, rest = lists[0], lists[1:]
        for letter in first:
            fn(rest, group + [letter], result)

    result = []
    fn(lists, result=result)
    return result


def display_multiple_faces(faces):
    if len(faces) == 1:
        faces[0].display()
        plt.show()
        return
    f, ax = plt.subplots(1, len(faces))
    for idx, face in enumerate(faces):
        plt.sca(ax[idx])
        face.display()
    plt.show()


def display_trajectory_faces(trajectory):
    transitions = trajectory.transitions
    states = [x[0] for x in transitions] + [trajectory.current_state]
    display_multiple_faces(states)


def read_fasta(filepath):
    all_seqs_dict = {}
    with open(filepath, 'r') as file:
        seq_id = None
        all_seqs = []
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                if len(all_seqs) > 0 and seq_id is not None:
                    all_seqs_dict[seq_id] = all_seqs
                seq_id = line
                all_seqs = []
            elif len(line) > 0:
                all_seqs.append(line)

        if len(all_seqs) > 0 and seq_id is not None:
            all_seqs_dict[seq_id] = all_seqs

    return all_seqs_dict


def sample_states_for_evaluation(
        env, state_nums, mutations_cutoff, max_duplicate_mutations, maintain_tree_shape=True,
        reference_trajectory=None):
    """
    :param mutations_cutoff: any total mutations higher than this cutoff will be considered duplicate
    :param max_duplicate_mutations: set a bound on the number of states with the same total mutations
    :param maintain_tree_shape: make sure we sample phylo trees with the identical tree topology
    :param reference_trajectory: if maintain_tree_shape option is enabled, the reference_trajectory argument
                                will provide the reference tree topology
    :return:
    """
    if reference_trajectory is None:
        reference_trajectory = env.sample(1)[0]

    # to obtain a diverse sample of total mutations (aka rewards)
    mutations_counter = defaultdict(lambda: 0)
    traj_mutations = reference_trajectory.current_state.subtrees[0].total_mutations
    if traj_mutations > mutations_cutoff:
        traj_mutations = mutations_cutoff
    mutations_counter[traj_mutations] += 1

    all_trajs = [reference_trajectory]
    nb_steps = 0
    nb_steps_max = 100000
    while len(all_trajs) < state_nums:
        nb_steps += 1
        if maintain_tree_shape:
            # constrain all sampled trees to have the same underlying topology
            traj = env.trajectory_permute_leaves(reference_trajectory)
        else:
            traj = env.sample(1)[0]
        traj_mutations = traj.current_state.subtrees[0].total_mutations
        if traj_mutations > mutations_cutoff:
            traj_mutations = mutations_cutoff
        if mutations_counter[traj_mutations] >= max_duplicate_mutations:
            if nb_steps <= nb_steps_max:
                continue
            else:
                print(f'Sampling step exceeds {nb_steps_max}, cannot satisfy \'max_duplicate_mutations\' requirements')
                max_duplicate_mutations = np.inf
        mutations_counter[traj_mutations] += 1
        all_trajs.append(traj)
    all_states = [traj.current_state for traj in all_trajs]
    return all_trajs, all_states, mutations_counter


def compute_paths_to_state(state, env):
    """
    count number of paths
    NOTE: for more than 50 species estimating this number is very very slow
    """
    if state.is_initial:
        return 1
    parents_states = env.get_parent_states(state)
    num_paths = 0
    for state, _ in parents_states:
        num_paths += compute_paths_to_state(state, env)
    return num_paths


def compute_trajectory_prob(generator, list_trajs, log_prob=False, batch_size=20000, scale_key=None):
    """
    compute for all trajs at once
    :param log_prob: whether return prob or log prob
    :return:
    """
    all_state, all_action, all_traj_idx = [], [], []
    nb_trajs = len(list_trajs)
    for traj_idx, trajectory in enumerate(list_trajs):  # typical amount: 1000*100 trajs
        for state, next_state, a, reward, done in trajectory.transitions:
            all_state.append(state)  # 1,600,000 states for 10 species
            all_action.append(a)
            all_traj_idx.append(traj_idx)

    all_action = torch.tensor(all_action).long().to(generator.all_device[-1])
    all_traj_idx = torch.tensor(all_traj_idx).long().to(generator.all_device[-1])
    traj_logpf = torch.zeros(nb_trajs, dtype=torch.float32).to(generator.all_device[-1])

    nb_states = len(all_state)  # 2 mins required to forward all trajectories
    with torch.no_grad():
        for i in range(0, nb_states, batch_size):
            if isinstance(generator, TBGFlowNetGenerator):
                logits, mask = generator(all_state[i: i + batch_size])
            else:
                logits, state_flows_logits, mask = generator(all_state[i: i + batch_size])
            logits = logits.masked_fill(mask, float('-inf'))
            all_log_pf = torch.nn.functional.log_softmax(logits, dim=1).gather(
                1, all_action[i: i + batch_size].unsqueeze(-1)).squeeze(-1)
            traj_logpf.scatter_add_(0, all_traj_idx[i: i + batch_size], all_log_pf)

    if log_prob:
        return traj_logpf
    else:
        return torch.exp(traj_logpf)


def display_trajectory(traj):
    for t in traj.transitions:
        t[0].display()
        print('-------------Apply Action {}-------'.format(t[2]))

    t[1].display()


def dummy_collate_fn(x):
    return x[0]


def load_sequences(sequences_path):
    # load sequences
    if sequences_path.endswith('.fa'):
        key_to_seqs_dict = read_fasta(sequences_path)
        # for now, only selecting the first set of sequences
        all_seqs = list(key_to_seqs_dict.values())[0]
    elif sequences_path.endswith('.pickle'):
        dict_species_seq = pickle.load(open(sequences_path, 'rb'))
        all_seqs = list(dict_species_seq.values())
    else:
        all_seqs = pickle.load(open(sequences_path, 'rb'))

    # todo, find better ways to address ? characters in MSA
    all_seqs = [seq if '?' not in seq else seq.replace('?', '-') for seq in all_seqs]

    return all_seqs


def generator_beam_search(generator, trajectories_num=10000):
    """
    trajectories_num is effectively beam width; beam search can be also used to sample trajectories
    but for now, we only use it for evaluation
    """
    env = generator.env
    state2input = generator.state2input
    all_traj_current_states = [env.get_initial_state()]
    all_traj_logp = torch.tensor([0.]).to(generator.all_device[-1])

    while np.any([not state.is_done for state in all_traj_current_states]):

        batch_state, batch_action, batch_idx = [], [], []
        for traj_idx, current_state in enumerate(all_traj_current_states):
            if not current_state.is_done:
                batch_state.append(current_state)
                batch_action.extend(range(len(current_state.subtrees)))
                batch_idx.extend([traj_idx] * len(current_state.subtrees))

        # shouldn't be any, but we keep it here for completeness and future reference
        completed_traj_idx_np = np.setdiff1d(np.arange(len(all_traj_current_states)), batch_idx)
        completed_traj_idx = torch.tensor(completed_traj_idx_np).to(generator.all_device[-1])
        batch_idx = torch.tensor(batch_idx).to(generator.all_device[-1])

        # ### below are new codes
        # # remove redundant states
        # batch_state_signature = [state.order_subtrees() for state in batch_state]
        # _, unique_idx, reconst_idx = \
        #     np.unique([signature[0] for signature in batch_state_signature], return_index=True, return_inverse=True)
        # batch_action_ordered = [batch_state_signature[i][1][action] for i, action in enumerate(batch_action)]
        # batch_action_ordered = torch.tensor(batch_action_ordered).to(generator.all_device[-1])
        # batch_unique_states = [batch_state[idx] for idx in unique_idx]
        # reconst_idx = torch.tensor(reconst_idx).to(generator.all_device[-1])

        with torch.no_grad():
            input_dict = state2input.states2inputs(batch_state)
            if isinstance(generator, TBGFlowNetGenerator):
                logits, mask = generator(input_dict)
            else:
                logits, state_flow_logits, mask = generator(input_dict)
            logits = logits.masked_fill(mask, float('-inf'))
            # logits = logits[reconst_idx]
            batch_log_pf = torch.nn.functional.log_softmax(logits, dim=1)
            batch_viable_log_pf = batch_log_pf.flatten()[~mask.flatten()]
            batch_viable_log_pf = all_traj_logp.gather(0, batch_idx) + batch_viable_log_pf

        all_log_pf = torch.cat([batch_viable_log_pf, all_traj_logp.gather(0, completed_traj_idx)])
        log_pf_sorted, log_pf_sorted_idx = all_log_pf.sort(descending=True)
        all_traj_logp = log_pf_sorted[:trajectories_num]

        all_state = []
        for idx in log_pf_sorted_idx[:trajectories_num].detach().cpu().numpy():
            if idx < len(batch_idx):
                traj_idx = batch_idx[idx].item()
                state = env.transition(all_traj_current_states[traj_idx], batch_action[idx])[1]
            else:
                traj_idx = completed_traj_idx_np[idx - len(batch_idx)]
                state = all_traj_current_states[traj_idx]
            all_state.append(state)

        all_traj_current_states = all_state

    return all_traj_current_states


def process_trajectories_tb(env, state2input, list_trajs):
    """
    preparing all needed tensor to accelerate loss computation
    NOTE: we no longer return list_trajs as it tend to cause memory leaks for pytorch dataloaders
    """
    parsimony_problem = env.parsimony_problem

    batch_state, batch_traj_idx = [], []
    batch_action, batch_pb_log, batch_log_reward = [], [], []
    if not parsimony_problem:
        batch_edge_action = []
    for i, traj in enumerate(list_trajs):
        batch_traj_idx.extend([i] * len(traj.transitions))
        parents_num = env.get_number_parents_trajectory(traj)
        pb_log_sum = np.log([1 / n for n in parents_num]).sum()
        batch_pb_log.append(pb_log_sum)
        batch_log_reward.append(traj.reward['log_reward'])
        for state, next_state, action, reward, done in traj.transitions:
            batch_state.append(state)
            if not parsimony_problem:
                batch_action.append(action['tree_action'])
                batch_edge_action.append(action['edge_action'])
            else:
                batch_action.append(action)

    input_dict = state2input.states2inputs(batch_state)
    input_dict['batch_pb_log'] = torch.tensor(np.array(batch_pb_log)).float()
    # NOTE: reward shaping may lead to an extreme range of reward
    input_dict['batch_log_reward'] = torch.tensor(batch_log_reward).float()
    input_dict['batch_action'] = torch.tensor(batch_action).long()
    if not parsimony_problem:
        batch_edge_action = np.array(batch_edge_action)
        if env.cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION in ['CATEGORICAL', 'CATEGORICAL_INDEPENDENT']:
            input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).long()
        else:
            input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).float()
    input_dict['batch_traj_idx'] = torch.tensor(batch_traj_idx).long()
    input_dict['batch_size'] = len(list_trajs)

    if 'pariwise_action_reverse_tensor' in input_dict:
        reverse_tensor = input_dict['pariwise_action_reverse_tensor']
        batch_action = input_dict['batch_action']
        batch_pairwise_action = reverse_tensor[torch.arange(len(batch_action)), batch_action]
        input_dict['batch_pairwise_action'] = batch_pairwise_action

    return input_dict


def log_linear_schedule(start, end, T, t):
    if t > T:
        return end

    start_log, end_log = np.log(start), np.log(end)
    v_log = start_log + (end_log - start_log) * t / T
    return np.exp(v_log)


def cascading_schedule(schedules, t):
    for T, value in schedules:
        if t < T:
            return value
    return schedules[-1][1]


def linear_schedule(start, end, T, t):
    if t > T:
        return end

    return start + (end - start) * t / T


def cosine_schedule(start, end, T, t):
    if t > T:
        return end

    v = end + (start - end) * (1 + math.cos(math.pi * t / T)) / 2
    return v


def schedule(start, end, T, t, type):
    assert type in ['LINEAR', 'COSINE', 'LOG_LINEAR']
    if type == 'LINEAR':
        v = linear_schedule(start, end, T, t)
    elif type == 'LOG_LINEAR':
        v = log_linear_schedule(start, end, T, t)
    else:
        v = cosine_schedule(start, end, T, t)
    return v


def correct_cfg_data(all_seqs, nb_gpus, cfg):
    """

    :param all_seqs:
    :param nb_gpus:
    :param cfg:
    :return:
    """
    seq_length = len(all_seqs[0])
    sequence_type = cfg.ENV.SEQUENCE_TYPE
    parsimony_problem = cfg.PARSIMONY_PROBLEM
    if parsimony_problem:
        vocab_size = {
            'DNA': 4,
            'RNA': 4,
            'DNA_WITH_GAP': 5,
            'RNA_WITH_GAP': 5
        }[sequence_type]
    else:
        vocab_size = 4

    cfg.ENV.EVOLUTION_MODEL.SEQUENCE_LENGTH = seq_length
    cfg.ENV.EVOLUTION_MODEL.VOCAB_SIZE = vocab_size
    cfg.GFN.MODEL.TRANSFORMER.SEQ_EMB.INPUT_SIZE = seq_length * vocab_size
    if cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION == 'CATEGORICAL':
        edges_cat_cfg = cfg.GFN.MODEL.EDGES_MODELING.CATEGORICAL
        if edges_cat_cfg.INDEPENDENT:
            bin_num = edges_cat_cfg.BINS
            edges_cat_cfg.HEAD.OUTPUT_SIZE = bin_num
            edges_cat_cfg.ROOT_EDGE_HEAD.OUTPUT_SIZE = bin_num
        else:
            bin_num = edges_cat_cfg.BINS
            edges_cat_cfg.HEAD.OUTPUT_SIZE = bin_num ** 2
            edges_cat_cfg.ROOT_EDGE_HEAD.OUTPUT_SIZE = bin_num
    if nb_gpus > 1:
        training_cfg = cfg.GFN.TRAINING_DATA_LOADER
        training_cfg.BEST_STATE_BATCH_SIZE = int(training_cfg.BEST_STATE_BATCH_SIZE / nb_gpus)
        training_cfg.GFN_BATCH_SIZE = int(training_cfg.GFN_BATCH_SIZE / nb_gpus)
        training_cfg.GFN_FIXED_SHAPE_BATCH_SIZE = int(training_cfg.GFN_FIXED_SHAPE_BATCH_SIZE / nb_gpus)
    return cfg
