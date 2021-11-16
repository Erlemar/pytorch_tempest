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
        num_cycles: float = 0.5,
        warmup_prop: float = 0.1,
        last_epoch: int = -1,
        epochs: int = 200,
        train_len: int = 6036000,
        n_folds: int = 5,
    ) -> None:
        len_train = train_len * (n_folds - 1) // n_folds
        self.num_warmup_steps = int(warmup_prop * epochs * len_train)
        self.num_training_steps = int(epochs * len_train)

        def lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch)
