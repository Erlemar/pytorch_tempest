from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.datasets.get_dataset import load_augs
from src.utils.ml_utils import stratified_group_k_fold
from src.utils.technical_utils import load_obj


class MelanomaDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        train = pd.read_csv(self.cfg.datamodule.train_path)
        train = train.rename(columns={'image_id': 'image_name'})
        train['image_name'] = train['image_name'] + '.jpg'

        # for fast training
        if self.cfg.training.debug:
            train, valid = train_test_split(train, test_size=0.1, random_state=self.cfg.training.seed)
            train = train[:1000]
            valid = valid[:1000]

        else:

            folds = list(
                stratified_group_k_fold(y=train['target'], groups=train['patient_id'], k=self.cfg.datamodule.n_folds)
            )
            train_idx, valid_idx = folds[self.cfg.datamodule.fold_n]

            valid = train.iloc[valid_idx]
            train = train.iloc[train_idx]

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        # initialize augmentations
        train_augs = load_augs(self.cfg['augmentation']['train']['augs'])
        valid_augs = load_augs(self.cfg['augmentation']['valid']['augs'])

        self.train_dataset = dataset_class(
            image_names=train['image_name'].values,
            transforms=train_augs,
            labels=train['target'].values,
            img_path=self.cfg.datamodule.path,
            mode='train',
            labels_to_ohe=False,
            n_classes=self.cfg.training.n_classes,
        )
        self.valid_dataset = dataset_class(
            image_names=valid['image_name'].values,
            transforms=valid_augs,
            labels=valid['target'].values,
            img_path=self.cfg.datamodule.path,
            mode='valid',
            labels_to_ohe=False,
            n_classes=self.cfg.training.n_classes,
        )

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

    def test_dataloader(self):
        return None
