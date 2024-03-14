import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, MixtureSameFamily, Independent
import numpy as np
from torch.distributions import Categorical
import copy
from src.model.mlp import MLP


# this models the log-branch length
class MixtureGaussianModel(nn.Module):

    def __init__(self, mixture_cfg, root_edge_mode=False):
        super().__init__()
        self.root_edge_mode = root_edge_mode
        mixture_cfg = copy.deepcopy(mixture_cfg)
        self.nb_components = mixture_cfg.NB_COMPONENTS
        self.soft_plus = nn.Softplus()

        # layout for ROOT_EDGE_HEAD output: [mixture_logits: nb_comp] - [mean: nb_comp] - [log_var: nb_comp]
        if root_edge_mode:
            mixture_cfg.HEAD.OUTPUT_SIZE = self.nb_components * 3
        elif mixture_cfg.EXPR_INDEPENDENT:
            # layout: [mixture_logits: nb_comp] - [mean: nb_comp * 2] - [log_co_var: nb_comp * 2]
            mixture_cfg.HEAD.OUTPUT_SIZE = self.nb_components * 5  # sampling lr from two independent distributions
        else:
            # layout: [mixture_logits: nb_comp] - [mean: nb_comp * 2] - [log_co_var: nb_comp * 4]
            mixture_cfg.HEAD.OUTPUT_SIZE = self.nb_components * 7  # sampling lr from the joint distribution

        self.mixture_cfg = mixture_cfg
        self.model = MLP(mixture_cfg.HEAD)

    def forward(self, input):
        logits = self.model(input)
        mixture_logits = logits[:, :self.nb_components]

        if self.root_edge_mode:
            mean = logits[:, self.nb_components: 2 * self.nb_components]
            mean = (torch.tanh(mean) + 1) / 2 * 10 - 10

            # trick, reasonable edge length are concentrated in a small range
            # model converges faster by sampling log edge length around -4.0 at the beginning
            # mean = mean - 4.0
            var_ = logits[:, 2 * self.nb_components:]
            if self.mixture_cfg.USE_SOFT_PLUS:
                var = self.soft_plus(var_) + 0.001
            else:
                var = var_.exp()
            dist = MixtureSameFamily(  # gaussian mixture model in 1D
                Categorical(logits=mixture_logits),
                Normal(mean, var),
            )
        elif self.mixture_cfg.EXPR_INDEPENDENT:
            mean = logits[:, self.nb_components: 3 * self.nb_components]. \
                reshape(-1, self.nb_components, 2)  # 2 is the event dimension
            # mean = mean - 4.0
            mean = (torch.tanh(mean) + 1) / 2 * 10 - 10
            var_ = logits[:, 3 * self.nb_components:]
            if self.mixture_cfg.USE_SOFT_PLUS:
                var = self.soft_plus(var_).reshape(-1, self.nb_components, 2) + 0.001
            else:
                var = var_.exp().reshape(-1, self.nb_components, 2)
            dist = MixtureSameFamily(
                Categorical(logits=mixture_logits),
                Independent(Normal(mean, var), 1),
            )
        else:
            mean = logits[:, self.nb_components: 3 * self.nb_components]. \
                reshape(-1, self.nb_components, 2)  # 2 is the event dimension
            mean = mean - 4.0
            covar_mat_ = logits[:, 3 * self.nb_components:].reshape(-1, self.nb_components, 2, 2)
            covar_mat_ = torch.matmul(covar_mat_, covar_mat_.transpose(2, 3))
            covar_mat = covar_mat_ + 2 * torch.eye(2).unsqueeze(0).unsqueeze(1).to(covar_mat_)  # positive definite
            dist = MixtureSameFamily(
                Categorical(logits=mixture_logits),
                MultivariateNormal(mean, covar_mat),
            )

        ret = {
            'logits': logits,
            'dist': dist
        }
        return ret


class EdgeModelContinuousMixture(nn.Module):

    def __init__(self, mixture_cfg):
        super(EdgeModelContinuousMixture, self).__init__()
        self.mixture_cfg = mixture_cfg
        self.nb_components = mixture_cfg.NB_COMPONENTS

        self.root_edge_model = MixtureGaussianModel(mixture_cfg, True)
        self.lr_model = MixtureGaussianModel(mixture_cfg, False)

    def forward(self, summary_reps, left_trees, right_trees, input_dict):

        random_spec = input_dict.get('random_spec', None)
        input_edge_actions = input_dict.get('input_edge_actions', None)

        # for now the representation is the concatenation of left and right tree
        rep = torch.cat([summary_reps, left_trees, right_trees], dim=1)
        batch_nb_seq = input_dict['batch_nb_seq']
        # in the new version, it is either all root edges or all first edges
        first_edges_flag = torch.any(batch_nb_seq > 2)

        # assert torch.all(root_edges_flag) or torch.all(first_edges_flag)
        ret = {}
        if first_edges_flag:
            first_edges_ret = self.lr_model(rep)
            edge_actions = self.sample(first_edges_ret, random_spec, first_edges_flag)
            ret['first_edges_actions'] = edge_actions
            ret['first_edges_ret'] = first_edges_ret
        else:
            root_edges_ret = self.root_edge_model(rep)
            edge_actions = self.sample(root_edges_ret, random_spec, first_edges_flag)
            ret['root_edges_actions'] = edge_actions
            ret['root_edges_ret'] = root_edges_ret

        if input_edge_actions is not None:
            # branch length to log branch length
            if first_edges_flag:
                edge_actions[:len(input_edge_actions)] = torch.log(input_edge_actions)
            else:
                edge_actions[:len(input_edge_actions)] = torch.log(input_edge_actions.sum(dim=-1))
        ret['log_paths_pf'] = self.compute_log_path_pf(ret, edge_actions, first_edges_flag)
        ret['edge_actions'] = torch.exp(edge_actions)
        return ret

    def sample(self, ret, random_spec, first_edges_flag):
        # note, this function samples log branch length

        dist = ret['dist']
        edge_action = dist.sample().clip(-10, 0)

        # todo, expr disabling exploration of the edge length
        return edge_action

    def compute_log_path_pf(self, ret, edge_actions, first_edges_flag):

        if first_edges_flag:
            dist = ret['first_edges_ret']['dist']
        else:
            dist = ret['root_edges_ret']['dist']

        # these are log_probs of the log branch length
        log_paths_pf_ = dist.log_prob(edge_actions)
        # change of variable to obtain log_probs of the branch length
        correction = edge_actions
        if first_edges_flag:
            correction = correction.sum(dim=1)
        log_paths_pf = log_paths_pf_ - correction

        return log_paths_pf
