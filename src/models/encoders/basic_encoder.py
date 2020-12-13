from typing import Union

import pretrainedmodels
import timm
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

from src.utils.ml_utils import freeze_until


class BasicEncoder(nn.Module):
    def __init__(
        self,
        arch: str = 'resnet18',
        source: str = 'pretrainedmodels',
        pretrained: Union[str, bool] = 'imagenet',
        n_layers: int = -2,
        freeze: bool = False,
        to_one_channel: bool = False,
        freeze_until_layer: str = None,
    ) -> None:
        """
        Initialize Encoder.

        Args:
            num_classes: the number of target classes, the size of the last layer's output
            arch: the name of the architecture form pretrainedmodels
            pretrained: the mode for pretrained model from pretrainedmodels
            n_layers: number of layers to keep
            freeze: to freeze model
            freeze_until: freeze until this layer. If None, then freeze all layers
        """
        super().__init__()
        if source == 'pretrainedmodels':
            net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
            self.output_dimension = list(net.children())[-1].in_features
        elif source == 'torchvision':
            net = torchvision.models.__dict__[arch](pretrained=pretrained)
            self.output_dimension = list(net.children())[-1].in_features
        elif source == 'timm':
            net = timm.create_model(arch, pretrained=pretrained)
            self.output_dimension = net.fc.in_features
        if source == 'efficientnet':
            net = EfficientNet.from_pretrained(arch)
            self.output_dimension = net._fc.in_features

        if freeze:
            freeze_until(net, freeze_until_layer)

        layers = list(net.children())[:n_layers]
        if to_one_channel:
            # https://www.kaggle.com/c/bengaliai-cv19/discussion/130311#745589
            # saving the weights of the first conv in w
            w = layers[0].weight
            # creating new Conv2d to accept 1 channel
            layers[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # substituting weights of newly created Conv2d with w from but we have to take mean
            # to go from  3 channel to 1
            layers[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x)

        return output
