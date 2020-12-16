from typing import Dict

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def mixup(batch: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    image = batch['image']
    target = batch['target']
    indices = torch.randperm(image.shape[0])
    shuffled_data = image[indices]
    shuffled_target = target[indices]
    # TODO compare sampling from numpy and pytorch. from torch.distributions import beta
    lam = np.random.beta(alpha, alpha)
    image = image * lam + shuffled_data * (1 - lam)

    return {'image': image, 'target': target, 'shuffled_target': shuffled_target, 'lam': lam}


def cutmix(batch: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    image = batch['image']
    target = batch['target']

    indices = torch.randperm(image.size(0))
    shuffled_data = image[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = image.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    image[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    return {'image': image, 'target': target, 'shuffled_target': shuffled_target, 'lam': lam}


class MixupCollator:
    """
    Mixup Collator

    This is a modified version of code from:
    https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/collators/mixup.py

    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = default_collate(batch)
        batch = mixup(batch, self.alpha)
        return batch


class CutMixCollator:
    """
    Cutmix Collator

    This is a modified version of code from:
    https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/collators/cutmix.py

    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch
