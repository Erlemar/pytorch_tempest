from omegaconf import DictConfig
from torch import nn

from src.utils.technical_utils import load_obj


class Net(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.encoder = load_obj(cfg.model.encoder.class_name)(**cfg.model.encoder.params)
        self.decoder = load_obj(cfg.model.decoder.class_name)(
            output_dimension=self.encoder.output_dimension, **cfg.model.decoder.params
        )
        self.loss = load_obj(cfg.loss.class_name)()

    def forward(self, x, targets):
        out = self.encoder(x)
        logits = self.decoder(out)
        loss = self.loss(logits, targets).view(1)
        return logits, loss
