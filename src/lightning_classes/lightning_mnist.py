from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

# class LightningMNISTClassifier(pl.LightningModule):
#
#     def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
#         super(LightningMNISTClassifier, self).__init__()
#         self.cfg = cfg
#         self.hparams: Dict[str, float] = hparams
#         # mnist images are (1, 28, 28) (channels, width, height)
#         self.layer_1 = torch.nn.Linear(28 * 28, 128)
#         self.layer_2 = torch.nn.Linear(128, 256)
#         self.layer_3 = torch.nn.Linear(256, 10)
#
#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#
#         # (b, 1, 28, 28) -> (b, 1*28*28)
#         x = x.view(batch_size, -1)
#
#         # layer 1 (b, 1*28*28) -> (b, 128)
#         x = self.layer_1(x)
#         x = torch.relu(x)
#
#         # layer 2 (b, 128) -> (b, 256)
#         x = self.layer_2(x)
#         x = torch.relu(x)
#
#         # layer 3 (b, 256) -> (b, 10)
#         x = self.layer_3(x)
#
#         # probability distribution over labels
#         x = torch.log_softmax(x, dim=1)
#
#         return x
#
#     def cross_entropy_loss(self, logits, labels):
#         return F.nll_loss(logits, labels)
#
#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         accuracy = torch.div(torch.eq(logits.argmax(1), y).sum(), float(y.size()[0]))
#
#         logs = {'train_loss': loss, 'train_acc': accuracy}
#         # print(logs)
#         return {'loss': loss, 'log': logs, 'progress_bar': logs}
#
#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         accuracy = torch.div(torch.eq(logits.argmax(1), y).sum(), float(y.size()[0]))
#         logs = {'val_loss': loss, 'val_acc': accuracy}
#
#         return {'val_loss': loss, 'val_acc': accuracy, 'log': logs, 'progress_bar': logs}
#
#     def validation_epoch_end(self, outputs):
#         # called at the end of the validation epoch
#         # outputs is an array with what you returned in validation_step for each batch
#         # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
#         logs = {'avg_val_loss': avg_loss, 'val_accuracy': accuracy}
#         return {'log': logs, self.cfg.training.metric: accuracy,
#                 'progress_bar': logs}
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
from src.utils.technical_utils import load_obj


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LightningMNISTClassifier, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        if not cfg.metric.params:
            self.metric = load_obj(cfg.metric.class_name)()
        else:
            self.metric = load_obj(cfg.metric.class_name)(**cfg.metric.params)

    def forward(self, x, targets, *args, **kwargs):
        return self.model(x, targets)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = torch.div(torch.eq(logits.argmax(1), y).sum(), float(y.size()[0]))

        logs = {'train_loss': loss, 'train_acc': accuracy}
        # print(logs)
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = torch.div(torch.eq(logits.argmax(1), y).sum(), float(y.size()[0]))
        logs = {'val_loss': loss, 'val_acc': accuracy}

        return {'val_loss': loss, 'val_acc': accuracy, 'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'val_accuracy': accuracy}
        return {'log': logs, self.cfg.training.metric: accuracy,
                'progress_bar': logs}

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.scheduler.step,
                              'monitor': self.cfg.scheduler.monitor}]
