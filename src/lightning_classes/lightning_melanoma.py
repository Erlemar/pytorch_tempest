from typing import Dict, Union, List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import numpy as np
from src.datasets.get_dataset import get_training_datasets
from src.utils.technical_utils import load_obj


class OLdLitMelanoma(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(OLdLitMelanoma, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)

    def forward(self, x, targets, *args, **kwargs):
        return self.model(x, targets)

    def prepare_data(self):
        datasets = get_training_datasets(self.cfg)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            shuffle=False,
        )

        return valid_loader

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

    def training_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        img, target = batch
        logits, loss = self(img, target)
        # result = pl.TrainResult(minimize=loss)
        # result.log('train_loss', loss, prog_bar=True)
        # result.logits = logits
        # result.target = target
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs, 'logits': logits, 'target': target}

    def training_epoch_end(self, outputs):
        y_true: List[np.array] = []
        y_pred: List[np.array] = []
        losses: List[np.array] = []
        for output in outputs:
            y_true.extend(output['target'].cpu().detach().numpy())
            y_pred.extend(torch.sigmoid(output['logits']).cpu().detach().numpy())
            losses.append(output['loss'].cpu().detach().numpy())
        # y_true = outputs.target.cpu().detach().numpy()
        # losses = outputs.train_loss.cpu().detach().numpy()
        # y_pred = torch.sigmoid(outputs.logits).cpu().detach().numpy()

        score = torch.tensor(roc_auc_score(np.array(y_true), np.array(y_pred)))
        loss = torch.tensor(np.sum(losses))

        logs = {'train_loss_total': loss, 'train_auc': score}
        # outputs.train_loss_total = loss
        # outputs.train_auc = score
        # return outputs
        return {'log': logs, 'progress_bar': logs}

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        img, target = batch
        logits, loss = self(img, target)
        # print(len(logits), len(target))
        # result = pl.EvalResult(loss)
        # result.log('val_loss', loss.mean(), prog_bar=True)
        # result.logits = logits
        # result.target = target
        logs = {'val_loss': loss}

        # return result
        # logs = {'val_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs, 'logits': logits, 'target': target}

    def validation_epoch_end(self, outputs):
        y_true: List[np.array] = []
        y_pred: List[np.array] = []
        losses: List[np.array] = []
        for output in outputs:
            y_true.extend(output['target'].cpu().detach().numpy())
            y_pred.extend(torch.sigmoid(output['logits']).cpu().detach().numpy())
            losses.append(output['loss'].cpu().detach().numpy())
        # print(outputs)
        # y_true = outputs.target.cpu().detach().numpy()
        # losses = outputs.val_loss.cpu().detach().numpy()
        # y_pred = torch.sigmoid(outputs.logits).cpu().detach().numpy()
        # print('!!', torch.cat(y_true).shape, torch.cat(y_pred).shape)
        score = torch.tensor(roc_auc_score(np.array(y_true), np.array(y_pred)))
        loss = torch.tensor(np.sum(losses))

        score = torch.tensor(1.0)
        logs = {'val_loss_total': loss, 'valid_auc': score}
        # outputs.log('val_loss_total', loss, prog_bar=True)
        # outputs.log('valid_auc', score, prog_bar=True)
        # outputs.log('main_score', score, prog_bar=True)
        # outputs.val_loss_total = loss
        # outputs.valid_auc = score
        # outputs.main_score = score
        # return outputs
        return {'val_loss': loss, 'log': logs, 'progress_bar': logs, 'main_score': score}


class LitMelanomaResult(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LitMelanomaResult, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)

    def forward(self, x, targets, *args, **kwargs):
        return self.model(x, targets)

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

    def training_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        img, target = batch
        logits, loss = self(img, target)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True, on_step=True)
        result.log('main_score', loss, prog_bar=True, on_step=True)
        # result.logits = logits
        return result

    def training_epoch_end(self, outputs):
        # y_true = outputs.target.cpu().detach().numpy()
        losses = outputs.train_loss.cpu().detach().numpy()
        # y_pred = torch.sigmoid(outputs.logits).cpu().detach().numpy()

        # score = torch.tensor(roc_auc_score(np.array(y_true), np.array(y_pred)))
        # loss = torch.tensor(np.sum(losses))
        # outputs.train_loss_total = loss
        # outputs.train_auc = score

        outputs.log('main_score1', torch.tensor(1.0), on_epoch=True, on_step=False)
        return outputs

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        img, target = batch
        logits, loss = self(img, target)
        # print(loss, loss.mean())
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(
            'val_loss', loss,
            prog_bar=True, on_step=True, on_epoch=False
        )
        result.log('main_score', torch.tensor(1.0), on_epoch=True, on_step=False)

        # result.logits = logits
        # result.target = target.float()

        return result

    def validation_epoch_end(self, outputs):
        # y_true = outputs.target.cpu().detach().numpy()
        losses = outputs.val_loss.cpu().detach().numpy()
        # y_pred = torch.sigmoid(outputs.logits).cpu().detach().numpy()

        # score = torch.tensor(roc_auc_score(np.array(y_true), np.array(y_pred)))
        loss = torch.tensor(np.sum(losses))

        score = torch.tensor(1.0)
        # outputs.log('val_loss_total', loss, prog_bar=True)
        outputs.log('valid_auc', score, on_epoch=True)
        outputs.log('main_score', score, on_epoch=True, on_step=False)
        # outputs.val_loss_total = loss
        # outputs.valid_auc = score
        # outputs.main_score = score
        return outputs


class LitMelanoma(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LitMelanoma, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        if not cfg.metric.params:
            self.metric = load_obj(cfg.metric.class_name)()
        else:
            self.metric = load_obj(cfg.metric.class_name)(**cfg.metric.params)

    def forward(self, x, targets, *args, **kwargs):
        return self.model(x, targets)

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

    def training_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        image = batch['image']
        target = batch['target']
        logits, loss = self(image, target)
        score = self.metric(logits.argmax(1), target)
        logs = {'train_loss': loss, f'train_{self.cfg.training.metric}': score}
        return {'loss': loss, 'log': logs, 'progress_bar': logs,
                'logits': logits, 'target': target, f'train_{self.cfg.training.metric}': score}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['target'] for x in outputs])
        y_pred = torch.cat([x['logits'] for x in outputs])
        score = self.metric(y_pred.argmax(1), y_true)

        # score = torch.tensor(1.0, device=self.device)

        logs = {'train_loss': avg_loss, f'train_{self.cfg.training.metric}': score}
        return {'log': logs, 'progress_bar': logs}

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        image = batch['image']
        target = batch['target']
        logits, loss = self(image, target)
        score = self.metric(logits.argmax(1), target)
        logs = {'valid_loss': loss, f'valid_{self.cfg.training.metric}': score}

        return {'loss': loss, 'log': logs, 'progress_bar': logs, 'logits': logits, 'target': target, f'valid_{self.cfg.training.metric}': score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['target'] for x in outputs])
        y_pred = torch.cat([x['logits'] for x in outputs])
        score = self.metric(y_pred.argmax(1), y_true)

        # score = torch.tensor(1.0, device=self.device)
        logs = {'valid_loss': avg_loss, f'valid_{self.cfg.training.metric}': score}
        return {'valid_loss': avg_loss, 'log': logs, 'progress_bar': logs, self.cfg.training.metric: score}
