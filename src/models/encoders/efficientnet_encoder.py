from efficientnet_pytorch import EfficientNet
from torch import nn

from src.utils.ml_utils import freeze_until


class EfficientNetEncoder(nn.Module):
    def __init__(self, arch: str = 'efficientnet-b0', freeze: bool = False, freeze_until_layer: str = None) -> None:
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

        self.net = EfficientNet.from_pretrained(arch)
        self.output_dimension = self.net._fc.in_features
        if freeze:
            freeze_until(self.net, freeze_until_layer)

    def forward(self, x):
        # TODO compare this and complex backbone
        output = self.net.extract_features(x)

        return output
