from typing import Dict, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj


class LitImageClassification(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LitImageClassification, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
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

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
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
        logs = {'train_loss': loss, f'train_{self.cfg.training.metric}': score}
        return {
            'loss': loss,
            'log': logs,
            'progress_bar': logs,
            'logits': logits,
            'target': target,
            f'train_{self.cfg.training.metric}': score,
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['target'] for x in outputs])
        y_pred = torch.cat([x['logits'] for x in outputs])
        score = self.metric(y_pred.argmax(1), y_true)

        # score = torch.tensor(1.0, device=self.device)

        logs = {'train_loss': avg_loss, f'train_{self.cfg.training.metric}': score}
        return {'log': logs, 'progress_bar': logs}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
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
        logs = {'valid_loss': loss, f'valid_{self.cfg.training.metric}': score}

        return {
            'loss': loss,
            'log': logs,
            'progress_bar': logs,
            'logits': logits,
            'target': target,
            f'valid_{self.cfg.training.metric}': score,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['target'] for x in outputs])
        y_pred = torch.cat([x['logits'] for x in outputs])
        score = self.metric(y_pred.argmax(1), y_true)

        # score = torch.tensor(1.0, device=self.device)
        logs = {'valid_loss': avg_loss, f'valid_{self.cfg.training.metric}': score, self.cfg.training.metric: score}
        return {'valid_loss': avg_loss, 'log': logs, 'progress_bar': logs}
