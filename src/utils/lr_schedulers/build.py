import torch
from src.utils.lr_schedulers.cosine_restart import CosineAnnealingWarmupRestarts
from src.utils.lr_schedulers.cosine import CosineAnnealingLR


def build_scheduler(optimizer, cfg_scheduler):
    assert cfg_scheduler.TYPE in ['COSINE_WITH_RESTART', 'LINEAR', 'COSINE', 'STEP']

    if cfg_scheduler.TYPE == 'COSINE_WITH_RESTART':
        caw_cfg = cfg_scheduler.COSINE_WITH_RESTART
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=caw_cfg.T0,
            max_lr=caw_cfg.LR_MAX,
            min_lr=caw_cfg.LR_MIN,
            cycle_mult=caw_cfg.CYCLE_MULTI
        )
    elif cfg_scheduler.TYPE == 'COSINE':
        cosine_cfg = cfg_scheduler.COSINE
        scheduler = CosineAnnealingLR(optimizer, cosine_cfg.T_MAX, cosine_cfg.LR_MAX, cosine_cfg.LR_MIN)
    elif cfg_scheduler.TYPE == 'LINEAR':
        linear_cfg = cfg_scheduler.LINEAR
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=linear_cfg.START_FACTOR
                                                      , end_factor=linear_cfg.END_FACTOR, total_iters=linear_cfg.T)
    else:
        step_cfg = cfg_scheduler.STEP
        gamma = step_cfg.GAMMA
        step_size = step_cfg.STEP_SIZE
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)
    return scheduler
