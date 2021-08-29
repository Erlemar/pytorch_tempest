import os

import pytest
import torch
from hydra import compose, initialize

from src.utils.technical_utils import load_obj


@pytest.mark.parametrize('opt_name', os.listdir('conf/optimizer'))
def test_optimizers(opt_name: str) -> None:
    optimizer_name = opt_name.split('.')[0]
    with initialize(config_path='../conf'):
        cfg = compose(config_name='config', overrides=[f'optimizer={optimizer_name}', 'private=default'])
        load_obj(cfg.optimizer.class_name)(torch.nn.Linear(1, 1).parameters(), **cfg.optimizer.params)
