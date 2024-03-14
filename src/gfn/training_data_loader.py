import os
import random
from heapq import heappush, heappushpop, heapify
from src.utils.utils import schedule
import pickle
import torch


class TrainingDataLoader(object):

    def __init__(self, cfg, env, rollout_worker, best_trees_path):

        self.cfg = cfg
        self.amp = self.cfg.AMP
        self.env = env
        self.rollout_worker = rollout_worker
        self.best_trees_path = best_trees_path
        loader_cfg = cfg.GFN.TRAINING_DATA_LOADER
        splits_num = loader_cfg.MINI_BATCH_SPLITS
        self.batch_size = int((loader_cfg.GFN_FIXED_SHAPE_BATCH_SIZE + loader_cfg.GFN_BATCH_SIZE +
                               loader_cfg.BEST_STATE_BATCH_SIZE) / splits_num)
        self.gfn_fixed_shape_batch_size = int(loader_cfg.GFN_FIXED_SHAPE_BATCH_SIZE / splits_num)
        self.gfn_batch_size = int(loader_cfg.GFN_BATCH_SIZE / splits_num)
        self.best_state_batch_size = int(loader_cfg.BEST_STATE_BATCH_SIZE / splits_num)
        self.steps_per_epoch = int(loader_cfg.STEPS_PER_EPOCH * splits_num)
        self.best_tree_buffer_size = loader_cfg.BEST_TREES_BUFFER_SIZE
        self.rollout_random_prob = loader_cfg.RANDOM_ACTION_PROB
        self.condition_on_scale = cfg.GFN.CONDITION_ON_SCALE

        self.best_trees_topology_only = cfg.GFN.TRAINING_DATA_LOADER.BEST_TREES_TOPOLOGY_ONLY
        if self.best_state_batch_size > 0:
            self.initialize_best_trees()

    def initialize_best_trees(self):

        if os.path.isfile(self.best_trees_path):
            self.best_trees = pickle.load(open(self.best_trees_path, 'rb'))
            if self.best_trees_topology_only:
                self.seen_trees_keys = {tree.tree_topology_id: tree for tree in self.best_trees}
            else:
                self.seen_trees_keys = {tree.signature: tree for tree in self.best_trees}
        else:
            self.best_trees = []
            self.seen_trees_keys = {}
            trajs = self.env.sample(1000, True)
            trees = sorted([x.current_state.subtrees[0] for x in trajs], key=lambda x: -x.log_score)
            for unrooted_tree in trees:
                signature = unrooted_tree.signature
                if signature not in self.seen_trees_keys:
                    self.seen_trees_keys[signature] = unrooted_tree
                    if len(self.best_trees) >= self.best_tree_buffer_size:
                        dropped_tree = heappushpop(self.best_trees, unrooted_tree)
                        del self.seen_trees_keys[dropped_tree.signature]
                    else:
                        heappush(self.best_trees, unrooted_tree)

    def generate_batch(self, generator, random_spec):

        input_actions_set = None
        if self.best_state_batch_size > 0:
            input_actions_set = []
            trees = random.choices(self.best_trees, k=self.best_state_batch_size)
            for t in trees:
                actions, _ = self.env.sample_backward_from_tree(t)
                input_actions_set.append(actions)

        if self.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                data, trajectories = self.rollout_worker.rollout(generator, self.batch_size, random_spec=random_spec,
                                                                 generate_full_trajectories=False,
                                                                 input_actions_set=input_actions_set)
        else:
            data, trajectories = self.rollout_worker.rollout(generator, self.batch_size, random_spec=random_spec,
                                                             generate_full_trajectories=False,
                                                             input_actions_set=input_actions_set)

        if self.best_state_batch_size > 0:
            min_best_scores = min(self.best_trees).log_score
            trees_indices = torch.where(data['log_scores'][self.best_state_batch_size:] > min_best_scores)[
                0].cpu().numpy()
            if len(trees_indices) > 0:
                trees_actions = [trajectories[self.best_state_batch_size:][idx].actions for idx in trees_indices]
                trees_log_scores = data['log_scores'][self.best_state_batch_size:][trees_indices]
                trees = self.env.batch_actions_to_trees(trees_actions, trees_log_scores)
                self.update_best_trees_buffer(trees)
        return data, trajectories

    def update_best_trees_buffer(self, trees):

        for tree in trees:
            signature = tree.signature
            if signature not in self.seen_trees_keys:
                self.seen_trees_keys[signature] = tree
                if len(self.best_trees) >= self.best_tree_buffer_size:
                    dropped_tree = heappushpop(self.best_trees, tree)
                    del self.seen_trees_keys[dropped_tree.signature]
                else:
                    heappush(self.best_trees, tree)

    def build_epoch_iterator(self, generator, exploration_specs):
        for step in range(self.steps_per_epoch):
            random_spec = self.generate_random_spec(exploration_specs, step)
            yield self.generate_batch(generator, random_spec), random_spec

    def update_best_trees(self, best_trees):

        self.best_trees = best_trees
        self.seen_trees_keys = {tree.signature: None for tree in self.best_trees}

    def generate_random_spec(self, exploration_specs, step):

        if exploration_specs is None:
            return None

        start_value = exploration_specs['start_value']
        end_value = exploration_specs['end_value']
        type = self.cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION.ANNEAL_TYPE
        value = schedule(start_value, end_value, self.steps_per_epoch, step, type=type)

        if exploration_specs['exploration_method'] == 'EPS_ANNEALING':
            random_spec = {
                'random_action_prob': value
            }
        else:
            random_spec = {
                'T': value
            }

        return random_spec
