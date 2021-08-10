from typing import Dict, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj


class LitImageClassification(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(LitImageClassification, self).__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        self.loss = load_obj(cfg.loss.class_name)()
        if not cfg.metric.params:
            self.metric = load_obj(cfg.metric.class_name)()
        else:
            self.metric = load_obj(cfg.metric.class_name)(**cfg.metric.params)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

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

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        )

    def training_step(self, batch, *args, **kwargs):  # type: ignore
        # TODO: one method for train/val step/epoch
        image = batch['image']
        logits = self(image)

        target = batch['target']
        shuffled_target = batch.get('shuffled_target')
        lam = batch.get('lam')
        if shuffled_target is not None:
            loss = self.loss(logits, (target, shuffled_target, lam)).view(1)
        else:
            loss = self.loss(logits, target)
        score = self.metric(logits.argmax(1), target)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f'train_{self.cfg.metric.metric_name}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def validation_step(self, batch, *args, **kwargs):  # type: ignore
        image = batch['image']
        logits = self(image)

        target = batch['target']
        shuffled_target = batch.get('shuffled_target')
        lam = batch.get('lam')
        if shuffled_target is not None:
            loss = self.loss(logits, (target, shuffled_target, lam), train=False).view(1)
        else:
            loss = self.loss(logits, target)
        score = self.metric(logits.argmax(1), target)

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{self.cfg.training.metric}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
