import os

import pytest
import torch
from hydra.experimental import compose, initialize_config_dir

from src.utils.technical_utils import load_obj

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/conf'


@pytest.mark.parametrize('opt_name', os.listdir(path + '/optimizer'))
def test_optimizers(opt_name: str, path: str = path) -> None:
    opt_name = opt_name.split('.')[0]
    with initialize_config_dir(config_dir=path):
        cfg = compose(config_name='config', overrides=[f'optimizer={opt_name}'])
        load_obj(cfg.optimizer.class_name)(torch.nn.Linear(1, 1).parameters(), **cfg.optimizer.params)
