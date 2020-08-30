import torch
from omegaconf import DictConfig
from torch import nn

from src.utils.technical_utils import load_obj


class MnistModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.loss = load_obj(cfg.loss.class_name)()

    def forward(self, x, targets):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        loss = self.loss(x, targets).view(1)
        return x, loss
