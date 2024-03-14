import types
import math
from functools import wraps
import warnings
from torch.optim.lr_scheduler import _LRScheduler
import weakref
from collections import Counter
from bisect import bisect_right


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, max_lr, min_lr, last_epoch=-1, verbose=False):
        self.T_max = T_max
        super().__init__(optimizer, last_epoch, verbose)

        nb_params_group = len(optimizer.param_groups)
        if isinstance(max_lr, float):
            self.base_max_lr = [max_lr] * nb_params_group  # first max learning rate
            self.max_lr = [max_lr] * nb_params_group  # max learning rate in the current cycle
        elif isinstance(max_lr, list):
            assert len(max_lr) == nb_params_group
            self.base_max_lr = max_lr  # first max learning rate
            self.max_lr = max_lr  # max learning rate in the current cycle
        else:
            raise ValueError(f'Unrecognized max_lr argument: {max_lr}')

        if isinstance(min_lr, float):
            self.min_lr = [min_lr] * nb_params_group
        elif isinstance(min_lr, list):
            assert len(min_lr) == nb_params_group
            self.min_lr = min_lr
        for param_group, lr in zip(self.optimizer.param_groups, self.max_lr):
            param_group['lr'] = lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        elif self._step_count == 1 and self.last_epoch > 0:
            return [min_lr + (max_lr - min_lr) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for max_lr, min_lr, group in
                    zip(self.max_lr, self.min_lr, self.optimizer.param_groups)]

        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (max_lr - min_lr) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for max_lr, min_lr, group in
                    zip(self.max_lr, self.min_lr, self.optimizer.param_groups)]

        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - min_lr) + min_lr
                for group, min_lr in zip(self.optimizer.param_groups, self.min_lr)]

    def _get_closed_form_lr(self):
        return [min_lr + (max_lr - min_lr) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for max_lr, min_lr in zip(self.max_lr, self.min_lr)]
