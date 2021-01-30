from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR


class LambdaLRConfig(LambdaLR):
    """
    Inherit LambdaLR so that it can be defined in config.
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_lambda: str, last_epoch: int = -1) -> None:
        lr_lambda_: Callable = eval(lr_lambda)
        self.optimizer = optimizer

        if not isinstance(lr_lambda_, list) and not isinstance(lr_lambda_, tuple):
            self.lr_lambdas = [lr_lambda_] * len(optimizer.param_groups)
        else:
            if len(lr_lambda_) != len(optimizer.param_groups):
                raise ValueError(f'Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda_)}')
            self.lr_lambdas = list(lr_lambda_)
        self.last_epoch = last_epoch
        super().__init__(optimizer, lr_lambda_, last_epoch)
