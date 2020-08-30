import torch.nn.functional as F
from pytorch_toolbelt.modules.activations import Mish
from torch import nn

from src.models.layers.layers import RMSNorm, GeM


class BasicDecoder(nn.Module):
    def __init__(self, pool_output_size: int = 2, n_classes: int = 1, output_dimension: int = 512) -> None:
        """
        Initialize Decoder.

        Args:
            pool_output_size: the size of the result feature map after adaptive pooling layer
            n_classes: n classes to output
            output_dimension: output dimension of encoder
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=pool_output_size)
        self.fc = nn.Linear(output_dimension * pool_output_size * pool_output_size, n_classes)

    def forward(self, x):
        x = self.pool(x)
        output = self.fc(x.view(x.size()[0], -1))

        return output


class LightHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.norm = RMSNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x


class CnnHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.5):
        super().__init__()

        self.dropout = dropout
        #
        # if MEMORY_EFFICIENT:
        #     MishClass = MemoryEfficientMish
        # else:
        MishClass = Mish

        self.mish1 = MishClass()
        self.conv = nn.Conv2d(in_features, in_features, 1, 1)
        self.norm = nn.BatchNorm2d(in_features)
        self.pool = GeM()

        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.mish2 = MishClass()
        self.rms_norm = RMSNorm(in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, num_classes)

    def forward(self, x):
        x = self.mish1(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.dropout)
        x = self.fc1(x)

        x = self.mish2(x)
        x = self.rms_norm(x)
        x = F.dropout(x, p=self.dropout / 2)
        x = self.fc2(x)
        return x


class HeavyHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # if MEMORY_EFFICIENT:
        #     MishClass = MemoryEfficientMish
        # else:
        MishClass = Mish

        self.bn1 = RMSNorm(in_features)
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.bn2 = RMSNorm(in_features // 2)
        self.mish = MishClass()
        self.fc2 = nn.Linear(in_features // 2, num_classes)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.mish(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x
