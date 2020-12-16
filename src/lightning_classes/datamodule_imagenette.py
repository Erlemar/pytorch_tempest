import glob
import os
from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.datasets.get_dataset import load_augs
from src.utils.technical_utils import load_obj


class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        mapping_dict = {
            'n01440764': 0,
            'n02102040': 1,
            'n02979186': 2,
            'n03000684': 3,
            'n03028079': 4,
            'n03394916': 5,
            'n03417042': 6,
            'n03425413': 7,
            'n03445777': 8,
            'n03888257': 9,
        }
        train_labels = []
        train_images = []
        for folder in glob.glob(f'{self.cfg.datamodule.path}/train/*'):
            class_name = os.path.basename(os.path.normpath(folder))
            for filename in glob.glob(f'{folder}/*'):
                train_labels.append(mapping_dict[class_name])
                train_images.append(filename)

        val_labels = []
        val_images = []

        for folder in glob.glob(f'{self.cfg.datamodule.path}/val/*'):
            class_name = os.path.basename(os.path.normpath(folder))
            for filename in glob.glob(f'{folder}/*'):
                val_labels.append(mapping_dict[class_name])
                val_images.append(filename)

        if self.cfg.training.debug:
            train_labels = train_labels[:1000]
            train_images = train_images[:1000]
            val_labels = val_labels[:1000]
            val_images = val_images[:1000]

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        # initialize augmentations
        train_augs = load_augs(self.cfg['augmentation']['train']['augs'])
        valid_augs = load_augs(self.cfg['augmentation']['valid']['augs'])

        self.train_dataset = dataset_class(
            image_names=train_images,
            labels=train_labels,
            transforms=train_augs,
            mode='train',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
        )
        self.valid_dataset = dataset_class(
            image_names=val_images,
            labels=val_labels,
            transforms=valid_augs,
            mode='valid',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
            collate_fn=load_obj(self.cfg.datamodule.collate_fn)() if self.cfg.datamodule.collate_fn else None,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
            collate_fn=load_obj(self.cfg.datamodule.collate_fn)() if self.cfg.datamodule.collate_fn else None,
        )

        return valid_loader

    def test_dataloader(self):
        return None
