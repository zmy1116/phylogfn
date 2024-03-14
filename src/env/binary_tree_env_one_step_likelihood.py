import torch
import random
import numpy as np
import itertools
from ete3 import TreeNode
from src.env.trajectory import Trajectory, SimpleTrajectory
from src.utils.evolution_model_torch import EvolutionModelTorch
from src.env.edges_env import build_edge_env
from copy import deepcopy

from torch import nn

CHARACTERS_MAPS = {
    'DNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'RNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'DNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    },
    'RNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    }
}


class PhyloTreeReward(object):

    def __init__(self, reward_cfg):
        self.C = reward_cfg.C
        assert reward_cfg.RESHAPE_METHOD in ['C-MUTATIONS', 'EXPONENTIAL']
        self.reshape_method = reward_cfg.RESHAPE_METHOD
        self.power = reward_cfg.POWER
        self.scale = reward_cfg.SCALE
        self.reward_exp_min = reward_cfg.EXP_MIN
        self.reward_exp_max = reward_cfg.EXP_MAX

    def exponential_reward(self, log_score):
        """
        :param log_score:
        :return:
        """
        log_reward = (self.C + log_score) / self.scale
        return log_reward

    def __call__(self, log_score):
        """
        compute reward
        """
        return self.exponential_reward(log_score)


class PhylogeneticTree(object):

    def __init__(self, ete_node, log_score, generate_signature=False):
        """

        :param ete_node:  ete root node
        :param log_score:
        :param generate_signature:
        """
        self.ete_node = ete_node
        self.log_score = log_score
        if generate_signature:
            tree_topology_id = self.ete_node.get_topology_id()
            self.tree_topology_id = tree_topology_id
            if log_score is not None:
                self.signature = tree_topology_id + '_{:.3f}'.format(log_score)
            else:
                self.signature = tree_topology_id + '_noscore'
        else:
            self.signature = 'partial_tree'

    def __str__(self):
        return self.ete_node.get_ascii(show_internal=True, attributes=["name", "dist"])

    def update_log_score(self, log_score):
        self.log_score = log_score
        if self.signature != 'partial_tree':
            self.signature = self.signature.split('_')[0] + '_{:.3f}'.format(log_score)

    def to_rooted_tree(self, node_idx, in_place=True):

        if not in_place:
            ete_node = deepcopy(self.ete_node)
        else:
            ete_node = self.ete_node
        outgroup_node = ete_node.search_nodes(name=node_idx)[0]
        ete_node.set_outgroup(outgroup_node)

        # update min_seq_idx of all nodes
        for node in ete_node.traverse("postorder"):
            if not node.is_leaf():
                node.min_seq_idx = min([x.min_seq_idx for x in node.children])

        return ete_node

    @classmethod
    def create_tree_node(cls, root_idx, children_nodes_data):
        new_node = TreeNode(name=root_idx, dist=0.0)
        new_node.sequences_indices = []
        is_leaf = len(children_nodes_data) == 0
        if is_leaf:
            new_node.min_seq_idx = new_node.name
            new_node.sequences_indices.append(root_idx)
        else:
            for tree, dist in children_nodes_data:
                node = tree.ete_node
                new_node.add_child(node, dist=dist)
                new_node.sequences_indices = new_node.sequences_indices + node.sequences_indices
            new_node.min_seq_idx = min([c.min_seq_idx for c in [x[0].ete_node for x in children_nodes_data]])
        return new_node

    @classmethod
    def create_last_unrooted_tree_node(cls, children_nodes_data):

        l_tree, l_length = children_nodes_data[0]
        r_tree, r_length = children_nodes_data[1]
        l_tree_node, r_tree_node = l_tree.ete_node, r_tree.ete_node
        dist = l_length + r_length
        if l_tree_node.name > r_tree_node.name:
            root_node = l_tree_node
            other_node = r_tree_node
        else:
            root_node = r_tree_node
            other_node = l_tree_node

        root_node.add_child(other_node, dist=dist)
        root_node.sequences_indices = l_tree_node.sequences_indices + r_tree_node.sequences_indices
        root_node.min_seq_idx = 0

        # set tree unrooted
        root_node.unroot()

        node_idx = root_node.name
        rooted_tree_root_idx = node_idx + 1

        # when converting unrooted tree back to rooted tree,  the original root node becomes a no name node
        # correct it with following

        # convert unrooted tree to rooted, set the new root node id 2n-1
        root_node.set_outgroup(other_node)
        root_node.name = rooted_tree_root_idx

        # get the no name node and give it id 2n - 2
        noname_node = root_node & ""
        noname_node.name = node_idx

        return root_node

    def __lt__(self, obj):
        return (obj.log_score - self.log_score) > 0.0001

    def __eq__(self, obj):
        return abs(self.log_score - obj.log_score) < 0.0001


class PhylogenticTreeState(object):

    def __init__(self, subtrees, max_root_idx, log_reward):
        self.subtrees = subtrees
        self.max_root_idx = max_root_idx
        self.is_done = (len(self.subtrees) == 1)
        self.log_reward = log_reward
        self.num_trees = len(self.subtrees)


class PhylogenticTreeEnv(nn.Module):

    def __init__(self, cfg, sequences):
        super(PhylogenticTreeEnv, self).__init__()
        self.cfg = cfg
        self.sequences = sequences
        self.reward_fn = PhyloTreeReward(cfg.ENV.REWARD)
        self.chars_dict = CHARACTERS_MAPS[cfg.ENV.SEQUENCE_TYPE]
        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences])
        self.seq_arrays = torch.nn.Parameter(torch.tensor(seq_arrays), requires_grad=False)
        self.evolution_model = EvolutionModelTorch(cfg.ENV.EVOLUTION_MODEL)
        # store for each number of trees, what are the possible combination of pairs
        self.tree_pairs_dict = {}
        self.action_indices_dict = {}
        for n in range(2, len(self.sequences) + 1):
            tree_pairs = list(itertools.combinations(list(np.arange(n)), 2))
            self.tree_pairs_dict[n] = tree_pairs
            self.action_indices_dict[n] = {pair: idx for idx, pair in enumerate(tree_pairs)}
        self.parsimony_problem = False
        self.edge_env = build_edge_env(cfg)

        self.seq_length = cfg.GFN.MODEL.SEQ_LEN
        self.normalize_tree_features = cfg.GFN.NORMALIZE_LIKELIHOOD
        self.pairwise_masking_data = {}
        for n in range(2, len(self.seq_arrays) + 1):
            self.pairwise_masking_data[n] = self.build_pairs_masks_mapping(n)
        self.type = cfg.ENV.ENVIRONMENT_TYPE

    def build_pairs_masks_mapping(self, max_num_seqs):
        rows, cols = np.triu_indices(max_num_seqs, k=1)
        data = []
        for index, (i, j) in enumerate(zip(rows, cols)):
            data.append([index, i, j])

        pairwise_valid_tensor = []
        for i in range(2, max_num_seqs + 1):
            value = i - 1
            pairwise_valid_tensor.append([x[1] <= value and x[2] <= value for x in data])
        pairwise_valid_tensor = np.array(pairwise_valid_tensor)

        x_list, y_list = np.where(pairwise_valid_tensor)
        actions_mapping_tensor = np.ones_like(pairwise_valid_tensor, dtype=int) * -1
        counter = 0
        current_x = 0
        actions_mapping_reverse_tensor = np.ones((len(pairwise_valid_tensor), len(rows)), dtype=int) * -1

        for x, y in zip(x_list, y_list):

            if x != current_x:
                current_x = x
                counter = 0

            actions_mapping_tensor[x, y] = counter
            actions_mapping_reverse_tensor[current_x, counter] = y
            counter += 1

        masks_tensor = ~ pairwise_valid_tensor
        return masks_tensor, actions_mapping_tensor, actions_mapping_reverse_tensor

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def get_initial_state(self):

        subtrees = []
        for idx in range(len(self.sequences)):
            node = TreeNode(name=idx, dist=0.0)
            node.min_seq_idx = node.name
            node.sequences_indices = [idx]
            tree = PhylogeneticTree(node, None)
            subtrees.append(tree)
        state = PhylogenticTreeState(subtrees, idx, None)
        return state

    def batch_apply_actions(self, actions, tree_features, states):
        """
        Apply actions
        if tree features is none, do not calculate score/ reward , just generate the tree objects
        if states is none, do not build the tree objects, just compute updated tree features
        :param actions:
        :param tree_features:
        :param states:
        :return:
        """
        # tree features and states cannot be both None
        update_features = tree_features is not None
        update_states = states is not None
        assert update_features or update_states
        if update_features:
            b, n, m, c = tree_features.shape
            # retrieve pair of trees features
            tree_action = [x['tree_action'] for x in actions]
            tree_pairs = self.retrieve_tree_pairs([n] * b, tree_action)
            left_trees_indices = [x[0] for x in tree_pairs]
            right_trees_indices = [x[1] for x in tree_pairs]
            left_features = tree_features[torch.arange(b), left_trees_indices]
            right_features = tree_features[torch.arange(b), right_trees_indices]

            # retrieve edge length
            edge_action = [x['edge_action'] for x in actions]
            at_root = n == 2
            edges = [self.edge_env.actions_2_edges(x, at_root=at_root) for x in edge_action]
            left_edge_lengths = [x[0] for x in edges]
            right_edge_lengths = [x[1] for x in edges]
            left_edge_lengths = torch.from_numpy(np.array(left_edge_lengths)).to(tree_features)
            right_edge_lengths = torch.from_numpy(np.array(right_edge_lengths)).to(tree_features)

            # compute the merged feature set
            data = [[left_features, left_edge_lengths], [right_features, right_edge_lengths]]
            merged_trees_features, log_scores = self.evolution_model.compute_partial_prob(data,
                                                                                          at_root)
            log_scores = log_scores.float()
            log_rewards = self.reward_fn(log_scores)

            # update trees features
            new_states_inputs_indices = torch.ones(b, n).bool()
            new_states_inputs_indices[torch.arange(b), right_trees_indices] = False
            new_trees_features = tree_features[new_states_inputs_indices]
            new_trees_features = new_trees_features.reshape(b, -1, m, c)
            new_trees_features[torch.arange(b), left_trees_indices] = merged_trees_features
        else:
            new_trees_features = None
            log_scores, log_rewards = None, None

        if update_states:
            new_states = []
            for idx, (state, action) in enumerate(zip(states, actions)):
                if update_features:
                    log_score, log_reward = log_scores[idx].item(), log_rewards[idx].item()
                else:
                    log_score, log_reward = None, None
                new_states.append(self.update_state(state, action, log_score, log_reward))
        else:
            new_states = None

        return new_states, new_trees_features, log_scores, log_rewards

    def update_state(self, state, action, log_score, log_reward):
        """
        update
        :param state:
        :param action:
        :param log_score:
        :param log_reward:
        :return:
        """
        tree_pair_action = action['tree_action']
        edge_pair_action = action['edge_action']
        l_length, r_length = self.edge_env.actions_2_edges(edge_pair_action, at_root=(state.num_trees == 2))
        tree_pairs = self.tree_pairs_dict[state.num_trees]
        i, j = tree_pairs[tree_pair_action]
        l_tree, r_tree = state.subtrees[i], state.subtrees[j]
        children_data = [[l_tree, l_length], [r_tree, r_length]]
        last_step = (len(state.subtrees) == 2)
        # if last step, merge the two trees such that the root node has 3 children
        if last_step:
            root_node = PhylogeneticTree.create_last_unrooted_tree_node(children_data)
            new_tree = PhylogeneticTree(root_node, log_score, generate_signature=True)
            state = PhylogenticTreeState([new_tree], state.max_root_idx, log_reward)
        else:
            root_idx = state.max_root_idx + 1
            root_node = PhylogeneticTree.create_tree_node(root_idx, children_data)
            new_tree = PhylogeneticTree(root_node, None)
            new_trees = []
            for idx in range(len(state.subtrees)):
                if idx not in (i, j):
                    new_trees.append(state.subtrees[idx])
                if idx == i:
                    new_trees.append(new_tree)
            state = PhylogenticTreeState(new_trees, root_idx, None)

        return state

    def build_tree_from_actions(self, actions, log_score):

        state = self.get_initial_state()
        for a in actions[:-1]:
            state = self.update_state(state, a, None, None)

        state = self.update_state(state, actions[-1], log_score, None)
        tree = state.subtrees[0]
        return tree

    def retrieve_tree_pairs(self, batch_nb_trees, batch_action):
        """
        Retrieve the pairs of trees to be joined
        :param batch_nb_trees: list of total number of trees per state
        :param batch_action:   list of actions to apply per state
        :return: list of pairs of trees
        """
        tree_pairs = []
        if type(batch_nb_trees) == torch.Tensor:
            batch_nb_trees = batch_nb_trees.cpu().numpy()
        if type(batch_action) == torch.Tensor:
            batch_action = batch_action.cpu().numpy()
        for num_trees, a in zip(batch_nb_trees, batch_action):
            tree_pair = self.tree_pairs_dict[num_trees][a]
            tree_pairs.append(tree_pair)
        return tree_pairs

    def compute_tree_log_score(self, ete_tree, with_noise):
        """
        compute log score of the ete tree object
        :param ete_tree:
        :return:
        """

        feature_dict = {}
        discrete_factor = 0
        for node in ete_tree.traverse("postorder"):
            if node.is_leaf():
                feature_dict[node.name] = self.seq_arrays[node.name].unsqueeze(0)
            else:
                children_nodes = node.children
                data = []
                if node.is_root() and with_noise:
                    edge_length = np.sum([x.dist for x in children_nodes])
                    root_noise = self.edge_env.generate_random_perturbation(edge_length, True)
                for c in children_nodes:
                    if with_noise:
                        if node.is_root():
                            noise = root_noise
                        else:
                            noise = self.edge_env.generate_random_perturbation(c.dist, False)
                    else:
                        noise = 0

                    data.append(
                        [feature_dict[c.name], torch.tensor([c.dist + noise]).to(self.seq_arrays)]
                    )
                feature, log_score = self.evolution_model.compute_partial_prob(data, node.is_root())
                feature_dict[node.name] = feature

                ## TODO CLEAN THIS ENTIRE MESS
                if with_noise:
                    if node.is_root():
                        edge_length = np.sum([x.dist for x in children_nodes])
                        if self.edge_env.bin_size_type == 'EQUAL_BIN_SIZE':
                            discrete_factor += np.log(self.edge_env.categorical_bin_size)
                        else:
                            a, b = self.edge_env.perturbation_ranges[edge_length]
                            discrete_factor += np.log(b - a)
                    else:
                        for c in children_nodes:
                            edge_length = c.dist
                            if self.edge_env.bin_size_type == 'EQUAL_BIN_SIZE':
                                discrete_factor += np.log(self.edge_env.categorical_bin_size)
                            else:
                                a, b = self.edge_env.perturbation_ranges[edge_length]
                                discrete_factor += np.log(b - a)

        return log_score, feature_dict, discrete_factor

    def sample_backward_from_tree(self, phylo_tree):
        """
        sample backward with uniform pb
        :return: list of actions, log_paths_pb
        """

        actions_list = []
        parents_num_list = []

        # last step
        action, parent_state_subtrees, parents_num = self.sample_backward_laststep(phylo_tree)
        actions_list.append(action)
        parents_num_list.append(parents_num)

        while True:
            action, parent_state_subtrees, parents_num = self.sample_backward_prevsteps(parent_state_subtrees)
            actions_list.append(action)
            parents_num_list.append(parents_num)

            no_leaf_trees = [x for x in parent_state_subtrees if not x.is_leaf()]
            if len(no_leaf_trees) == 0:
                break
        actions_list = actions_list[::-1]
        parents_num_list = parents_num_list[::-1]
        log_paths_pb = - np.log(np.array(parents_num_list))
        return actions_list, log_paths_pb

    def sample_backward_laststep(self, phylo_tree):
        """
        helper fn for sample_backward_from_tree: last step
        :param phylo_tree: PhyloTree object
        :return:
        """
        n = len(self.seq_arrays)
        # IMPORTANT THAT IT'S 0 TO 2N-3 TO
        node_idx = random.randint(0, 2 * n - 3)
        rooted_tree = phylo_tree.to_rooted_tree(node_idx)
        children = rooted_tree.children
        children = sorted(children, key=lambda x: x.min_seq_idx)
        edges = [x.dist for x in children]
        edge_action = self.edge_env.edges_2_actions(edges[0], edges[1], at_root=True)
        action = {'tree_action': 0, 'edge_action': edge_action}
        return action, children, 2 * n - 3

    def sample_backward_prevsteps(self, trees):
        """

        :param trees: list of ete trees
        :return:
        """
        non_leaf_idx = [idx for idx, x in enumerate(trees) if not x.is_leaf()]

        split_idx = random.choice(non_leaf_idx)
        split_trees = trees[split_idx].children
        split_trees = sorted(split_trees, key=lambda x: x.min_seq_idx)

        new_trees = trees[:split_idx] + trees[split_idx + 1:] + split_trees
        new_trees = sorted(new_trees, key=lambda x: x.min_seq_idx)
        tree_pos = {x.min_seq_idx: idx for idx, x in enumerate(new_trees)}
        tree_pair = (tree_pos[split_trees[0].min_seq_idx], tree_pos[split_trees[1].min_seq_idx])
        tree_action = self.action_indices_dict[len(new_trees)][tree_pair]
        edges = [x.dist for x in split_trees]
        edge_action = self.edge_env.edges_2_actions(edges[0], edges[1], at_root=False)
        action = {'tree_action': tree_action, 'edge_action': edge_action}

        return action, new_trees, len(non_leaf_idx)

    def prepare_rollout_inputs(self, tree_features, actions, random_spec):

        assert len(tree_features.shape) == 4
        # normalize per site
        if self.normalize_tree_features:
            inputs = tree_features / tree_features.sum(-1, keepdims=True)
            inputs = inputs.float()
            inputs[torch.isnan(inputs)] = 0.25
        else:
            inputs = tree_features.float()

        # flatten m x c -> mc
        b, n, _, _ = inputs.shape
        inputs = inputs.reshape(b, n, -1)
        batch_nb_seq = np.array([n] * b)
        mask_tensor, action_mapping_tensor, actions_mapping_reverse_tensor = self.pairwise_masking_data[n]
        mask_tensor = torch.tensor(mask_tensor[batch_nb_seq - 2])
        action_mapping_tensor = torch.tensor(action_mapping_tensor[batch_nb_seq - 2])
        actions_mapping_reverse_tensor = torch.tensor(actions_mapping_reverse_tensor[batch_nb_seq - 2])
        batch_nb_seq = torch.tensor(batch_nb_seq).long()

        input_dict = {
            'batch_input': inputs,
            'batch_nb_seq': batch_nb_seq.to(inputs.device),
            'pairwise_mask_tensor': mask_tensor.to(inputs.device),
            'pairwise_action_tensor': action_mapping_tensor.to(inputs.device),
            'pariwise_action_reverse_tensor': actions_mapping_reverse_tensor.to(inputs.device),
            'return_tree_reps': True,
            'batch_traj_idx': torch.arange(b).to(inputs.device),
            'batch_size': b,
            'random_spec': random_spec
        }
        if actions is not None:
            tree_actions = torch.tensor([x['tree_action'] for x in actions if 'tree_action' in x]).to(inputs.device)
            edges_action = torch.tensor([x['edge_action'] for x in actions if 'edge_action' in x]).to(inputs.device)
            input_dict['input_tree_actions'] = tree_actions
            input_dict['input_edge_actions'] = edges_action

        return input_dict

    def generate_random_trajectory(self):

        state = self.get_initial_state()
        tree_features = self.seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(1, 1, 1, 1)
        trajectory = Trajectory(state)
        while not state.is_done:
            n = len(state.subtrees)
            actions = list(range(int(n * (n - 1) / 2)))
            tree_action_pair = random.sample(actions, 1)[0]
            edge_action_pair = self.edge_env.generate_random_actions(at_root=(n == 2))
            a = {'tree_action': tree_action_pair, 'edge_action': edge_action_pair}

            new_states, new_trees_features, log_scores, log_rewards = self.batch_apply_actions([a], tree_features,
                                                                                               [state])
            trajectory.update(new_states[0], a, log_rewards[0].item(), new_states[0].is_done)
            state = new_states[0]
            tree_features = new_trees_features
        return trajectory

    def sample(self, num_trajs, generate_full_trajectory):

        if generate_full_trajectory:
            states = [self.get_initial_state() for _ in range(num_trajs)]
            trajectories = [Trajectory(x) for x in states]
        else:
            states = None
            trajectories = [SimpleTrajectory() for _ in range(num_trajs)]
        tree_features = self.seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(num_trajs, 1, 1, 1)

        while tree_features.shape[1] > 1:
            batch_actions = []
            for _ in range(num_trajs):
                n = tree_features.shape[1]
                actions = list(range(int(n * (n - 1) / 2)))
                tree_action_pair = random.sample(actions, 1)[0]
                edge_action_pair = self.edge_env.generate_random_actions(at_root=(n == 2))
                a = {'tree_action': tree_action_pair, 'edge_action': edge_action_pair}
                batch_actions.append(a)
            new_states, tree_features, log_scores, log_rewards = self.batch_apply_actions(batch_actions, tree_features,
                                                                                          states)
            if generate_full_trajectory:
                for a, s, r, traj in zip(batch_actions, new_states, log_rewards, trajectories):
                    traj.update(s, a, r.item(), s.is_done)
            else:
                for a, r, traj in zip(batch_actions, log_rewards, trajectories):
                    traj.update(a, r.item())

            states = new_states
        return trajectories

    def actions_to_trajectory(self, actions):

        state = self.get_initial_state()
        tree_features = self.seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(1, 1, 1, 1)
        trajectory = Trajectory(state)
        for a in actions:
            new_states, tree_features, log_scores, log_rewards = self.batch_apply_actions([a], tree_features,
                                                                                          [state])
            trajectory.update(new_states[0], a, log_rewards[0].item(), new_states[0].is_done)
            state = new_states[0]
        return trajectory

    def batch_actions_to_trees(self, batch_actions, batch_log_scores):

        states = [self.get_initial_state() for _ in batch_actions]
        for idx in range(len(batch_actions[0])):
            actions = [x[idx] for x in batch_actions]
            states, tree_features, log_scores, log_rewards = self.batch_apply_actions(actions, None,
                                                                                      states)

        trees = [x.subtrees[0] for x in states]
        for t, score in zip(trees, batch_log_scores):
            t.update_log_score(score.item())

        return trees
