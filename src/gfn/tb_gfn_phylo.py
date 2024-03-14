import math
import torch
import numpy as np
from src.model.edges_model.build import build_edge_model
from src.model.tree_topologies_model.build import build_tree_model
from src.utils.lr_schedulers.build import build_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0)
}


class FullModel(torch.nn.Module):

    def __init__(self, tree_model, edges_model, _Z):
        super(FullModel, self).__init__()
        self.tree_model = tree_model
        self.edges_model = edges_model
        self._Z = _Z


class TBGFlowNetGenerator(torch.nn.Module):
    def __init__(self, gfn_cfg, env, device_rank, ddp):

        super().__init__()
        self.gfn_model_cfg = gfn_model_cfg = gfn_cfg.MODEL
        self.apply_fast_Z = gfn_model_cfg.TB_FAST_Z
        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.scale_set = gfn_cfg.SCALES_SET
        self.parsimony_problem = env.parsimony_problem
        self.env = env

        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.scale_set = gfn_cfg.SCALES_SET
        self.parsimony_problem = env.parsimony_problem

        # load model
        tree_model = build_tree_model(gfn_cfg, env.type)
        if not self.parsimony_problem:
            edges_model = build_edge_model(gfn_cfg)
        else:
            edges_model = None

        trajs = env.sample(1000, generate_full_trajectory=False)
        self.max_reward_seen = np.max([x.log_reward for x in trajs])
        if self.condition_on_scale:
            init_Z = (self.max_reward_seen / torch.tensor(np.array(self.scale_set))).reshape(1, -1) / 256
            _Z = torch.nn.Parameter(  # in log
                torch.ones(len(self.scale_set), 256, device=self.all_device[0])
                * init_Z, requires_grad=True)
            update_z = True
        else:
            if gfn_model_cfg.Z_PARTITION_INIT == -1:
                init_Z = self.max_reward_seen
            else:
                init_Z = gfn_model_cfg.Z_PARTITION_INIT
            update_z = gfn_model_cfg.UPDATE_Z
            _Z = torch.nn.Parameter(  # in log
                torch.ones(256, device=device_rank) * init_Z / 256, requires_grad=update_z
            )

        # create full model in case need to wrap everytihng in DDP
        self.full_model = FullModel(tree_model, edges_model, _Z)
        self.full_model.to(device_rank)

        if ddp:
            self.full_model = DDP(self.fullresp_model, device_ids=[device_rank])

        if ddp:
            self.tree_model = self.full_model.module.tree_model
            self.edges_model = self.full_model.module.edges_model
            self._Z = self.full_model.model._Z
        else:
            self.tree_model = self.full_model.tree_model
            self.edges_model = self.full_model.edges_model
            self._Z = self.full_model._Z

        # Z and other model parts use different learning rate
        params = list(self.tree_model.parameters())
        if self.edges_model is not None:
            params = params + list(self.edges_model.parameters())
        params = [
            {'params': params, 'lr': gfn_model_cfg.LR_MODEL}
        ]
        if update_z:
            params = params + [{'params': [self._Z], 'lr': gfn_model_cfg.LR_Z}]

        # gradient clipping exclude the Z part
        self.gradient_clipping_params = list(self.tree_model.parameters()) + list(self.edges_model.parameters())
        self.grad_clip = gfn_model_cfg.GRAD_CLIP

        # optimizer
        self.opt = torch.optim.Adam(
            params,
            weight_decay=gfn_model_cfg.L2_REG, betas=(0.9, 0.999), amsgrad=True)

        # lr scheduler
        if gfn_cfg.MODEL.USE_LR_SCHEDULER:
            self.scheduler = build_scheduler(self.opt, gfn_cfg.MODEL.LR_SCHEDULER)
        else:
            self.scheduler = None

        # loss function
        self.loss_fn = LOSS_FN[gfn_model_cfg.LOSS_FN]

        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in self.gradient_clipping_params]))

        # scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler()

        # var to accumulate loss
        self.loss = 0

    def save(self, path):
        # we need to include optimizers state_dict as well
        torch.save({
            'generator_state_dict': self.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
        }, path)

    def load(self, path):
        # loading all state dicts
        all_state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(all_state_dict['generator_state_dict'])
        self.opt.load_state_dict(all_state_dict['opt_state_dict'])
        if not self.gfn_model_cfg.USE_LR_SCHEDULER:

            if self.opt.param_groups[0]['lr'] != self.gfn_model_cfg.LR_MODEL:
                self.opt.param_groups[0]['lr'] = self.gfn_model_cfg.LR_MODEL

            if self.opt.param_groups[1]['lr'] != self.gfn_model_cfg.LR_Z:
                self.opt.param_groups[1]['lr'] = self.gfn_model_cfg.LR_Z

    def train_step(self, input_batch):
        self.opt.zero_grad()
        loss = self.get_loss(input_batch)  # compute loss for all trajectories at once
        loss.backward()
        info = {'grad_norm': self.grad_norm(self.model),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self.model),
                'loss': loss.detach().cpu().numpy().tolist()}
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return info

    def compute_log_Z(self, scale_key=None):

        if self.condition_on_scale:
            log_z = self._Z[scale_key].sum(-1)
            return log_z
        else:
            return self._Z.sum()

    def Z(self, scale_key=None):
        return np.exp(self.log_Z(scale_key))

    def log_Z(self, scale_key=None):

        with torch.no_grad():
            log_z = self.compute_log_Z(scale_key)
            if len(log_z) > 0:
                log_z = log_z[0]
        return log_z.item()

    def get_loss_from_raw_inputs(self, input_dict):
        ret = self(input_dict)
        log_paths_pf = ret['log_paths_pf']
        batch_size = input_dict['batch_size']
        log_pf = torch.zeros(batch_size, dtype=torch.float32).to(log_paths_pf.device). \
            scatter_add_(0, input_dict['batch_traj_idx'], log_paths_pf)
        log_paths_pb = ret['log_paths_pb']
        log_pb = torch.zeros(batch_size, dtype=torch.float32).to(log_paths_pf.device). \
            scatter_add_(0, input_dict['batch_traj_idx'], log_paths_pb)
        log_reward = input_dict['batch_log_reward']
        log_z = self.compute_log_Z(input_dict.get('scale_key')).reshape(-1).to(log_pf.device)
        forward_value = log_z + log_pf
        backward_value = log_reward + log_pb
        loss = self.loss_fn(forward_value, backward_value)
        return loss

    def get_loss_from_rollout_outputs(self, rollout_outputs):

        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        log_rewards = rollout_outputs['log_rewards']

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)
        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb
        loss = self.loss_fn(forward_value, backward_value)
        return loss

    def forward(self, input_dict):
        """
        assume all input states have the same input/output dimension and the same state type
        """
        trees_ret = self.tree_model(**input_dict)
        tree_actions = trees_ret['tree_actions'].cpu().numpy()
        tree_pairs = self.env.retrieve_tree_pairs(input_dict['batch_nb_seq'], tree_actions)
        trees_ret['tree_pairs'] = tree_pairs
        ret = {
            'trees_ret': trees_ret
        }

        # for likelihood problem continue the computation
        if not self.parsimony_problem:
            left_trees_indices = [x[0] for x in tree_pairs]
            right_trees_indices = [x[1] for x in tree_pairs]
            n = len(tree_pairs)
            left_trees_reps = trees_ret['trees_reps'][torch.arange(n), left_trees_indices]
            right_trees_reps = trees_ret['trees_reps'][torch.arange(n), right_trees_indices]
            edges_ret = self.edges_model(trees_ret['summary_reps'], left_trees_reps, right_trees_reps, input_dict)
            ret['edges_ret'] = edges_ret

        ret['log_paths_pf'] = ret['edges_ret']['log_paths_pf'] + ret['trees_ret']['log_paths_pf']
        return ret

    def get_weighted_loss_from_rollout_outputs(self, rollout_outputs, weights):

        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        log_rewards = rollout_outputs['log_rewards']

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)
        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        err = torch.abs(forward_value - backward_value).detach()
        # loss = self.loss_fn(forward_value, backward_value)
        loss = torch.mean((forward_value - backward_value) ** 2 * weights)

        return loss, err

    def update_model_per(self, rollout_outputs, weights):
        loss, err = self.get_weighted_loss_from_rollout_outputs(rollout_outputs, weights)
        loss.backward()
        self.loss += loss
        info = {'grad_norm': self.grad_norm(self),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0
        return info, err.cpu().numpy()

    def accumulate_loss(self, rollout_outputs, factor=1.0):
        """
        for now only take input from rollout outputs
        :param rollout_outputs:
        :param factor:
        :return:
        """
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        loss = (loss / factor)
        loss.backward()
        self.loss += loss

    def update_model(self):

        info = {'grad_norm': self.grad_norm(self),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0

        return info

    def accumulate_loss_amp(self, rollout_outputs, factor=1.0):

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = self.get_loss_from_rollout_outputs(rollout_outputs)
            loss = (loss / factor)
        self.scaler.scale(loss).backward()
        self.loss += loss

    def update_model_amp(self):
        info = {'grad_norm': self.grad_norm(self),
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self.param_norm(self),
                'loss': self.loss.detach().cpu().numpy().tolist()}

        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self.loss = 0

        return info
