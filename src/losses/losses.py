from typing import Tuple

import torch
import torch.functional as F
from torch import nn


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric (kaggle ventilator)
    """

    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae


class MAE(nn.Module):
    def __call__(self, preds, y, u_out):
        # print(preds.shape, y.shape)
        return torch.nn.L1Loss(preds, y).mean()


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
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss


class MixupLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/mixup.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss
