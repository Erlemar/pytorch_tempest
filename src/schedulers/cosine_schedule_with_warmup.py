import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineScheduleWithWarmupConfig(LambdaLR):
    """
    Inherit LambdaLR so that it can be defined in config.
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_with_hard_restarts_schedule_with_warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch)
