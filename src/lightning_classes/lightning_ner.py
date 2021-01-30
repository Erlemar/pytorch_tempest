from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj


class LitNER(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig, tag_to_idx: Dict):
        super(LitNER, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.tag_to_idx = tag_to_idx
        self.model = load_obj(cfg.model.class_name)(
            embeddings_dim=cfg.datamodule.embeddings_dim, tag_to_idx=tag_to_idx, **cfg.model.params
        )
        if not cfg.metric.params:
            self.metric = load_obj(cfg.metric.class_name)()
        else:
            self.metric = load_obj(cfg.metric.class_name)(**cfg.metric.params)

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
        f1_score = self.metric(tag_seq, labels)
        log = {'f1_score': f1_score, 'loss': loss.item()}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        embeds, lens, labels = batch
        embeds = self._vectorizer(embeds)
        score, tag_seq, loss = self.model(embeds, lens, labels)
        labels = labels.flatten()
        labels = labels[labels != self.tag_to_idx['PAD']]
        f1_score = self.metric(tag_seq, labels)

        return {'val_loss': loss.item(), 'tag_seq': tag_seq, 'labels': labels, 'step_f1': f1_score.item()}

    def validation_epoch_end(self, outputs):
        avg_loss = np.stack([x['val_loss'] for x in outputs]).mean()
        f1_mean = np.stack([x['step_f1'] for x in outputs]).mean()
        y_true = torch.cat([x['labels'] for x in outputs])
        y_pred = torch.cat([x['tag_seq'] for x in outputs])

        f1_score = self.metric(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
        tensorboard_logs = {'main_score': f1_score, 'f1_mean': f1_mean}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
