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

#
# get_arch = lambda: nn.Sequential(
#     nn.Sequential(*list(resnet50(pretrained=True).children())[:6]),
#     nn.Sequential(*list(resnet50(pretrained=True).children())[6:-2]),
#     create_head(4096, 264),
#     nn.Sigmoid()
# )
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(*list(torchvision.models.resnet34(True).children())[:-2])
#         self.classifier = nn.Sequential(*[
#             nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),
#             nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),
#             nn.Linear(512, len(classes))
#         ])
#
#     def forward(self, x):
#         bs, im_num, ch, y_dim, x_dim = x.shape
#         x = self.cnn(x.view(-1, ch, y_dim, x_dim))
#         x = x.mean((2, 3))
#         x = self.classifier(x)
#         x = x.view(bs, im_num, -1)
#         x = lme_pool(x)
#         return x
#
