from typing import Dict

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from src.datasets.mnist_dataset import MnistDataset


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg

    def prepare_data(self):
        # download
        MnistDataset(self.data_dir, train=True, download=True)
        MnistDataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MnistDataset(self.data_dir, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            if self.cfg.training.debug:
                self.mnist_train, _ = random_split(self.mnist_train, [1000, 54000])
                self.mnist_val, _ = random_split(self.mnist_val, [1000, 4000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
