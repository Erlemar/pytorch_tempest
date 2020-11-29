import torch
import torch.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size()[0], -1)


def lme_pool(x, alpha=1.0):  # log-mean-exp pool
    """
    Pooling lme.
    alpha -> approximates maxpool, alpha -> 0 approximates mean pool
    Args:
        x:
        alpha:

    Returns:
        result of pooling
    """
    T = x.shape[1]
    return 1 / alpha * torch.log(1 / T * torch.exp(alpha * x).sum(1))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps=self.eps)'


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.

    # https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/modules/rms_norm.html#RMSNorm
    """

    def __init__(self, dimension: int, epsilon: float = 1e-8, is_bias: bool = False):
        """
        Args:
            dimension (int): the dimension of the layer output to normalize
            epsilon (float): an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias (bool): a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = torch.sqrt(torch.mean(x ** 2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if self.is_bias:
            return self.scale * x_norm + self.bias
        return self.scale * x_norm


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: float):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x


# def softmax_windows(self, x):
#     """
#     https://towardsdatascience.com/swap-softmax-weighted-average-pooling-70977a69791b
#     Take image `x`, with dimension (height, width).
#     Convert to (2, 2, height*width/4): tile image into 2x2 blocks
#     Take softmax over each of these blocks
#     Convert softmax'd image back to (height, width)
#
#     Usage:
#     self.pool = nn.AvgPool2d(2, 2)
#     x = self.pool(self.softmax_windows(a) * a)
#     """
#     x_strided = einops.rearrange(x, 'b c (h hs) (w ws) -> b c h w (hs ws)', hs=2, ws=2)
#     x_softmax_windows = F.softmax(x_strided, dim=-1)
#     x_strided_softmax = einops.rearrange(x_softmax_windows, 'b c h w (hs ws) -> b c (h hs) (w ws)', hs=2, ws=2)
#
#     return x_strided_softmax
