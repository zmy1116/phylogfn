import torch
import torch.nn as nn
from src.model.mlp import MLP
from torch.distributions import Categorical
import numpy as np

"""
Categorical 1 : left and right edges model
"""


class EdgesModelCategorical(nn.Module):

    def __init__(self, edge_cat_cfg):
        super(EdgesModelCategorical, self).__init__()
        # model last step edge
        self.root_edge_model = MLP(edge_cat_cfg.ROOT_EDGE_HEAD)

        # model for first n-1 steps left and right edges
        self.lr_model = MLP(edge_cat_cfg.HEAD)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.lr_actions_list = np.arange(edge_cat_cfg.HEAD.OUTPUT_SIZE)
        self.edges_independent = edge_cat_cfg.INDEPENDENT
        self.root_edge_actions_list = np.arange(edge_cat_cfg.ROOT_EDGE_HEAD.OUTPUT_SIZE)

    #
    def forward(self, summary_reps, left_trees, right_trees, input_dict):

        random_spec = input_dict.get('random_spec', None)
        input_edge_actions = input_dict.get('input_edge_actions', None)

        # for now the representation is the concatenation of left and right tree
        rep = torch.cat([summary_reps, left_trees, right_trees], dim=1)
        batch_nb_seq = input_dict['batch_nb_seq']
        first_edges_flag = (batch_nb_seq[0] > 2).item()
        ret = {}
        if first_edges_flag > 0:
            if self.edges_independent:
                l_edges_reps = torch.cat([summary_reps, left_trees], dim=1)
                r_edges_reps = torch.cat([summary_reps, right_trees], dim=1)
                l_edges_logits = self.lr_model(l_edges_reps)
                r_edges_logits = self.lr_model(r_edges_reps)
                first_edges_ret = {
                    'l_logits': l_edges_logits,
                    'r_logits': r_edges_logits
                }
                l_actions = self.sample(l_edges_logits, self.lr_actions_list, random_spec)
                r_actions = self.sample(r_edges_logits, self.lr_actions_list, random_spec)
                actions = torch.cat([l_actions[None], r_actions[None]], dim=0).T
                ret['first_edges_actions'] = actions
                edge_actions = actions
                ret['first_edges_ret'] = first_edges_ret
            else:
                first_edges_logits = self.lr_model(rep)
                first_edges_ret = {
                    'logits': first_edges_logits
                }
                actions = self.sample(first_edges_logits, self.lr_actions_list, random_spec)
                ret['first_edges_actions'] = actions
                edge_actions = actions
                ret['first_edges_ret'] = first_edges_ret
        else:
            root_edges_logits = self.root_edge_model(rep)
            root_edges_ret = {
                'logits': root_edges_logits
            }
            actions = self.sample(root_edges_logits, self.root_edge_actions_list, random_spec)
            ret['root_edges_actions'] = actions
            edge_actions = actions
            ret['root_edges_ret'] = root_edges_ret

        if input_edge_actions is not None:
            edge_actions[:len(input_edge_actions)] = input_edge_actions
        ret['edge_actions'] = edge_actions
        log_paths_pf = self.compute_log_path_pf(ret, input_dict['batch_nb_seq'], edge_actions)
        ret['log_paths_pf'] = log_paths_pf
        return ret

    def sample(self, logits, actions_list, random_spec):
        if random_spec is None:
            random_spec = {
                'random_action_prob': 0.0
            }
        if 'random_action_prob' in random_spec:
            random_p = random_spec['random_action_prob']
            distribution = Categorical(logits=logits)
            edge_action = distribution.sample()
            if random_p > 0:
                batch_size = edge_action.shape[0]
                rand_flag = (torch.empty(batch_size).uniform_(0, 1)) <= random_p
                rand_num = rand_flag.sum().item()
                if rand_num > 0:
                    rand_actions = torch.tensor(np.random.choice(actions_list, rand_num)).to(edge_action)
                    edge_action[rand_flag] = rand_actions
        else:
            T = random_spec['T']
            distribution = Categorical(logits=logits / T)
            edge_action = distribution.sample()
        return edge_action

    def compute_log_path_pf(self, ret, batch_nb_seq, edge_actions):

        root_edges_flag = batch_nb_seq == 2
        first_edges_flag = batch_nb_seq > 2
        log_paths_pf = torch.zeros(len(batch_nb_seq)).to(edge_actions.device)

        if first_edges_flag.sum().item() > 0:
            if self.edges_independent:
                first_edges_ret = ret['first_edges_ret']
                first_edges_actions = edge_actions[first_edges_flag]
                log_p_l = self.logsoftmax(first_edges_ret['l_logits'])
                log_p_r = self.logsoftmax(first_edges_ret['r_logits'])
                pf_l = log_p_l[torch.arange(len(first_edges_actions)), first_edges_actions[:, 0]]
                pf_r = log_p_r[torch.arange(len(first_edges_actions)), first_edges_actions[:, 1]]
                pf = pf_r + pf_l
                log_paths_pf[first_edges_flag] = log_paths_pf[first_edges_flag] + pf
            else:
                first_edges_ret = ret['first_edges_ret']
                first_edges_actions = edge_actions[first_edges_flag]
                log_p = self.logsoftmax(first_edges_ret['logits'])
                pf = log_p[torch.arange(len(first_edges_actions)), first_edges_actions]
                log_paths_pf[first_edges_flag] = log_paths_pf[first_edges_flag] + pf

        if root_edges_flag.sum().item() > 0:
            root_edges_ret = ret['root_edges_ret']
            log_p = self.logsoftmax(root_edges_ret['logits'])
            root_edges_actions = edge_actions[root_edges_flag]
            pf = log_p[torch.arange(len(root_edges_actions)), root_edges_actions]
            log_paths_pf[root_edges_flag] = log_paths_pf[root_edges_flag] + pf

        return log_paths_pf