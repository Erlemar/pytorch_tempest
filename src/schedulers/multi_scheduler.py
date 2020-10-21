import torch
from torch.optim.lr_scheduler import MultiplicativeLR


class MultiplicativeLRConfig(MultiplicativeLR):
    """
    Inherit MultiplicativeLR so that it can be defined in config.
    Reference: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiplicativeLR
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_lambda: str, last_epoch=-1):
        lr_lambda = eval(lr_lambda)
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f'Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}')
            self.lr_lambdas = list(lr_lambda)

        self.last_epoch = last_epoch
        super(MultiplicativeLR, self).__init__(optimizer, last_epoch)
