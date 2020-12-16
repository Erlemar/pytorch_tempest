from typing import Tuple

import torch
import torch.functional as F
from torch import nn


class DenseCrossEntropy(nn.Module):
    # Taken from: https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


class CutMixLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/cutmix.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        targets1, targets2, lam = targets
        if train:
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets1)
        return loss


class MixupLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/mixup.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        targets1, targets2, lam = targets
        if train:
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets1)
        return loss
