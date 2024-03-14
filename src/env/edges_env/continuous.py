import itertools
import numpy as np
import random
import math


class EdgeEnvContinuous(object):

    def __init__(self, edge_continuous_cfg):
        self.edge_continuous_cfg = edge_continuous_cfg

    def generate_random_actions_lr(self):
        edge_action_ = np.random.uniform(-10, 0, 2)
        edge_action = np.exp(edge_action_)
        return float(edge_action[0]), float(edge_action[1])

    def generate_random_actions_root(self):
        edge_action_ = np.random.uniform(-10, 0)
        edge_action = np.exp(edge_action_)
        return float(edge_action)

    def actions_2_edges(self, action, **other_input):
        if other_input['at_root']:
            return float(action / 2), float(action / 2)
        else:
            return float(action[0]), float(action[1])

    def edges_2_actions(self, left_length, right_length, **other_input):

        return left_length, right_length

    def generate_random_actions(self, **other_input):
        if other_input['at_root']:
            return self.generate_random_actions_root()
        else:
            return self.generate_random_actions_lr()
