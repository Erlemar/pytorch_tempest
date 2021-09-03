from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj


class LitNER(pl.LightningModule):
    def __init__(self, cfg: DictConfig, tag_to_idx: Dict):
        super(LitNER, self).__init__()
        self.cfg = cfg
        self.tag_to_idx = tag_to_idx
        self.model = load_obj(cfg.model.class_name)(
            embeddings_dim=cfg.datamodule.embeddings_dim, tag_to_idx=tag_to_idx, **cfg.model.params
        )
        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(self.cfg.metric.metric.class_name)(
                    **cfg.metric.metric.params
                )
            }
        )
        if 'other_metrics' in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics.update({metric.metric_name: load_obj(metric.class_name)(**metric.params)})

    def forward(self, x, lens, *args, **kwargs):
        return self.model(x, lens)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        )

    def training_step(self, batch, batch_idx):
        embeds, lens, labels = batch
        # transform tokens to embeddings
        embeds = self._vectorizer(embeds)
        score, tag_seq, loss = self.model(embeds, lens, labels)
        labels = labels.flatten()
        labels = labels[labels != self.tag_to_idx['PAD']]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.metrics:
            score = self.metrics[metric](tag_seq, labels)
            self.log(f'train_{metric}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeds, lens, labels = batch
        embeds = self._vectorizer(embeds)
        score, tag_seq, loss = self.model(embeds, lens, labels)
        labels = labels.flatten()
        labels = labels[labels != self.tag_to_idx['PAD']]

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.metrics:
            score = self.metrics[metric](tag_seq, labels)
            self.log(f'valid_{metric}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
