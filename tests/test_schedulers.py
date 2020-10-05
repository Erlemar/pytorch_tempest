import os

import pytest
import torch
from hydra.experimental import compose, initialize_config_dir

from src.utils.technical_utils import load_obj

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/conf'


@pytest.mark.parametrize('sch_name', os.listdir(path + '/scheduler'))
def test_schedulers(sch_name: str, path: str = path) -> None:
    scheduler_name = sch_name.split('.')[0]
    with initialize_config_dir(config_dir=path):
        cfg = compose(config_name='config', overrides=[f'scheduler={scheduler_name}', 'optimizer=sgd'])
        optimizer = load_obj(cfg.optimizer.class_name)(torch.nn.Linear(1, 1).parameters(), **cfg.optimizer.params)
        load_obj(cfg.scheduler.class_name)(optimizer, **cfg.scheduler.params)
