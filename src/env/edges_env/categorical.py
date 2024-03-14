import itertools
import numpy as np
import random
import math


class EdgeEnvCategorical(object):

    def __init__(self, edge_cat_cfg):
        self.categorical_bin_size = edge_cat_cfg.BIN_SIZE
        self.bin_size_type = edge_cat_cfg.BIN_SIZE_TYPE
        assert self.bin_size_type in ['EQUAL_BIN_SIZE', 'EQUAL_LOG_BIN_SIZE', 'EQUAL_EXPONENTIAL_FACTOR']
        self.categorical_bins = edge_cat_cfg.BINS
        self.edges_independent = edge_cat_cfg.INDEPENDENT
        if self.bin_size_type != 'EQUAL_BIN_SIZE':

            if self.bin_size_type == 'EQUAL_LOG_BIN_SIZE':
                log_bin_size_min, log_bin_size_max = edge_cat_cfg.LOG_BIN_SIZE_RANGE
                delta = (np.log(log_bin_size_max) - np.log(log_bin_size_min)) / self.categorical_bins
                self.categorical_bin_sizes = np.exp(
                    np.arange(np.log(log_bin_size_min), np.log(log_bin_size_max) + delta, delta))
            else:
                self.exp_factor = edge_cat_cfg.BIN_SIZE_EXP_FACTOR
                self.categorical_bin_sizes = np.round(
                    np.array([0.001 * self.exp_factor ** x for x in range(self.categorical_bins)]), 5)
            self.perturbation_ranges = {}
            bin_sizes = self.categorical_bin_sizes
            for idx, b in enumerate(bin_sizes):
                if idx == 0:
                    min_r = -(bin_sizes[idx + 1] - b) / 2
                else:
                    min_r = -(b - bin_sizes[idx - 1]) / 2

                if idx == len(bin_sizes) - 1:
                    max_r = (b - bin_sizes[idx - 1]) / 2
                else:
                    max_r = (bin_sizes[idx + 1] - b) / 2

                self.perturbation_ranges[b] = [min_r, max_r]
            self.categorical_bin_sizes_map = {v: idx for idx, v in enumerate(self.categorical_bin_sizes)}

        self.lr_actions_pairs = list(
            itertools.product(np.arange(self.categorical_bins), np.arange(self.categorical_bins)))
        self.lr_actions_pairs_indices = {
            pair: idx for idx, pair in enumerate(self.lr_actions_pairs)
        }
        self.max_edge_length = float(self.categorical_bins * self.categorical_bin_size)

    def generate_random_perturbation(self, edge_length, is_root):

        if self.bin_size_type != 'EQUAL_BIN_SIZE':
            a, b = self.perturbation_ranges[round(edge_length, 5)]
            noise = np.random.uniform(a, b)
        else:
            bin_size = self.categorical_bin_size
            noise = np.random.uniform(-0.5 * bin_size, 0.5 * bin_size)

        if is_root:
            noise = noise /2
        return noise

    def lr_actions_2_edges(self, edge_action):
        """

        :param edge_action: one action in N^2 of (l,r) pairs OR pair of edges length actions
        :return:
        """
        if self.edges_independent:
            left_length_action, right_length_action = edge_action
        else:
            left_length_action, right_length_action = self.lr_actions_pairs[edge_action]
        if self.bin_size_type == 'EQUAL_BIN_SIZE':
            left_length = (left_length_action + 1) * self.categorical_bin_size
            right_length = (right_length_action + 1) * self.categorical_bin_size
        else:
            left_length = self.categorical_bin_sizes[left_length_action]
            right_length = self.categorical_bin_sizes[right_length_action]
        return left_length, right_length

    def root_edge_actions_2_edges(self, edge_action):
        """
        calculate l r edge length at root level, since at the root we only care about total length
        since at the root level we only care about the total length, return l/2 for left and right
        :param edge_action: one action in N
        :return:
        """
        if self.bin_size_type == 'EQUAL_BIN_SIZE':
            edge_length = (1 + edge_action) * self.categorical_bin_size
        else:
            edge_length = self.categorical_bin_sizes[edge_action]
        return edge_length / 2, edge_length / 2

    def lr_edges_2_actions(self, left_length, right_length):

        if self.bin_size_type == 'EQUAL_BIN_SIZE':
            left_length_action = int(round((left_length / self.categorical_bin_size))) - 1
            right_length_action = int(round((right_length / self.categorical_bin_size))) - 1

        else:
            left_length_action = self.categorical_bin_sizes_map[round(left_length, 5)]
            right_length_action = self.categorical_bin_sizes_map[round(right_length, 5)]

        if self.edges_independent:
            action = [left_length_action, right_length_action]
        else:
            action = self.lr_actions_pairs_indices[(left_length_action, right_length_action)]
        return action

    def root_edge_2_actions(self, left_length, right_length):
        if self.bin_size_type == 'EQUAL_BIN_SIZE':
            length = left_length + right_length
            action = int(round((length / self.categorical_bin_size))) - 1
        else:
            length = left_length + right_length
            action = self.categorical_bin_sizes_map[round(length, 5)]
        return action

    def generate_random_actions_lr(self):
        if self.edges_independent:
            left_length_action = random.randint(0, self.categorical_bins - 1)
            right_length_action = random.randint(0, self.categorical_bins - 1)
            edge_action = [left_length_action, right_length_action]
        else:
            edge_action = random.randint(0, len(self.lr_actions_pairs) - 1)
        return edge_action

    def generate_random_actions_root(self):
        edge_action = random.randint(0, self.categorical_bins - 1)
        return edge_action

    def actions_2_edges(self, action, **other_input):

        if other_input['at_root']:
            return self.root_edge_actions_2_edges(action)
        else:
            return self.lr_actions_2_edges(action)

    def edges_2_actions(self, left_length, right_length, **other_input):

        if other_input['at_root']:
            action = self.root_edge_2_actions(left_length, right_length)
        else:
            action = self.lr_edges_2_actions(left_length, right_length)
        return action

    def generate_random_actions(self, **other_input):
        if other_input['at_root']:
            return self.generate_random_actions_root()
        else:
            return self.generate_random_actions_lr()