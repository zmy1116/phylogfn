import numpy as np
import torch
from torch import nn


def decompJC(symm=False):
    # pA = pG = pC = pT = .25
    pden = np.array([.25, .25, .25, .25])
    rate_matrix_JC = 1.0 / 3 * np.ones((4, 4))
    for i in range(4):
        rate_matrix_JC[i, i] = -1.0

    if not symm:
        D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
        U_JC_inv = np.linalg.inv(U_JC)
    else:
        D_JC, W_JC = np.linalg.eigh(
            np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0 / pden))))
        U_JC = np.dot(np.diag(np.sqrt(1.0 / pden)), W_JC)
        U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))

    return D_JC, U_JC, U_JC_inv, rate_matrix_JC


class EvolutionModelTorch(nn.Module):

    def __init__(self, evolution_cfg):
        super().__init__()
        self.D, self.U, self.U_inv, self.rateM = decompJC()

        self.prior_lambda = evolution_cfg.PRIOR_LAMBDA
        self.compute_prior = evolution_cfg.COMPUTE_PRIOR
        self.vocab_size = evolution_cfg.VOCAB_SIZE
        self.D = torch.nn.Parameter(torch.from_numpy(self.D), requires_grad=False)
        self.U = torch.nn.Parameter(torch.from_numpy(self.U), requires_grad=False)
        self.U_inv = torch.nn.Parameter(torch.from_numpy(self.U_inv), requires_grad=False)
        self.rateM = torch.nn.Parameter(torch.from_numpy(self.rateM), requires_grad=False)
        self.m = evolution_cfg.SEQUENCE_LENGTH

    def get_transition_matrices(self, edge_lengths):
        branch_D = torch.einsum("i,j->ij", (edge_lengths, self.D))
        transition_matrices = torch.matmul(
            torch.einsum("ij,kj->kij", (self.U, torch.exp(branch_D))), self.U_inv).clamp(0.0)
        return transition_matrices

    def compute_log_prior_p(self, edge_lengths):
        """
        compute prior p using exponential pdf
        :param edge_lengths:
        :return:
        """
        log_prior_p = np.log(self.prior_lambda) - self.prior_lambda * edge_lengths
        return log_prior_p

    def compute_partial_prob(self, data, rooted_tree_top):

        if rooted_tree_top:
            assert len(data) == 2

        joint_out = 1
        for feature, branch_length in data:
            transition_matrices = self.get_transition_matrices(branch_length)
            out = torch.einsum('bmv,bvc->bmc', (feature, transition_matrices))
            joint_out = joint_out * out

        if rooted_tree_top:
            edge_lengths = data[0][1] + data[1][1]
            log_branch_prior = self.compute_log_prior_p(edge_lengths)
            prior_portion = torch.exp(log_branch_prior / self.m).reshape(-1, 1, 1)
        else:
            prior_portion = 1
            for _, branch_length in data:
                log_branch_prior = self.compute_log_prior_p(branch_length)
                prior_portion = prior_portion * torch.exp(log_branch_prior / self.m).reshape(-1, 1, 1)
        joint_out = joint_out * prior_portion

        log_p = torch.sum(torch.log(torch.sum(joint_out / 4, -1)), -1)
        return joint_out, log_p
