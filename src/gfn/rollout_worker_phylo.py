import numpy as np
import torch
from src.env.trajectory import Trajectory, SimpleTrajectory


class RolloutWorker:

    def __init__(self, env):
        self.env = env

    # TODO SCALES
    def rollout(self, generator, episodes, scales=None, random_spec=None, generate_full_trajectories=False,
                input_actions_set=None):

        seq_arrays = self.env.seq_arrays
        n, m, c = seq_arrays.shape

        # store all actions
        all_actions = []

        # store all parents number for log paths pbs
        all_num_parents = []

        # store all log paths pfs
        all_log_paths_pf = []

        # initial tree features
        tree_features = seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(episodes, 1, 1, 1)

        # store whether each tree of state has children
        has_children = torch.zeros(episodes, n).bool()

        if generate_full_trajectories:
            states = [self.env.get_initial_state() for _ in range(episodes)]
            trajectories = [Trajectory(s) for s in states]
        else:
            states = None
            trajectories = [SimpleTrajectory() for _ in range(episodes)]

        step = 0
        while tree_features.shape[1] > 1:

            # prepare input dict
            input_actions = [x[step] for x in input_actions_set] if input_actions_set is not None else None
            input_dict = self.env.prepare_rollout_inputs(tree_features, input_actions, random_spec)

            # forward inference
            ret = generator(input_dict)
            trees_ret = ret['trees_ret']

            # tree pair actions
            tree_actions = trees_ret['tree_actions'].detach().cpu().numpy()
            actions = [{'tree_action': x} for x in tree_actions]
            if not self.env.parsimony_problem:
                edges_ret = ret['edges_ret']
                edge_actions = edges_ret['edge_actions'].detach().cpu().numpy()
                for idx, a in enumerate(actions):
                    a['edge_action'] = edge_actions[idx]

            all_actions.append(actions)
            states, new_tree_features, log_scores, log_rewards = self.env.batch_apply_actions(actions,
                                                                                              tree_features,
                                                                                              states)
            if generate_full_trajectories:
                for a, s, r, traj in zip(actions, states, log_rewards, trajectories):
                    traj.update(s, a, r.item(), s.is_done)
            else:
                for a, r, traj in zip(actions, log_rewards, trajectories):
                    traj.update(a, r.item())

            # collect num of possible parents to calculate pb
            b, n, _, _ = tree_features.shape
            left_trees_indices = [x[0] for x in trees_ret['tree_pairs']]
            right_trees_indices = [x[1] for x in trees_ret['tree_pairs']]
            new_states_inputs_indices = torch.ones(b, n).bool()
            new_states_inputs_indices[torch.arange(b), right_trees_indices] = False
            has_children = has_children[new_states_inputs_indices]
            has_children = has_children.reshape(b, -1)
            has_children[torch.arange(b), left_trees_indices] = 1
            num_parents = has_children.sum(-1)
            all_num_parents.append(num_parents)

            # add log paths pf
            all_log_paths_pf.append(ret['log_paths_pf'])
            tree_features = new_tree_features
            step += 1

        all_log_paths_pf = torch.stack(all_log_paths_pf).T
        all_num_parents = torch.stack(all_num_parents).T
        all_num_parents[:, -1] = 2 * seq_arrays.shape[0] - 3
        log_paths_pb = -torch.log(all_num_parents).to(all_log_paths_pf)
        data = {
            'log_paths_pf': all_log_paths_pf,
            'log_paths_pb': log_paths_pb,
            'log_rewards': log_rewards,
            'log_scores': log_scores,
            'random_spec': random_spec,
            'scales': scales
        }
        return data, trajectories
