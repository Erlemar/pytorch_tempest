import os

import pytest
import torch
from hydra import compose, initialize

from src.utils.technical_utils import load_obj


@pytest.mark.parametrize('sch_name', os.listdir('conf/scheduler'))
def test_schedulers(sch_name: str) -> None:
    scheduler_name = sch_name.split('.')[0]
    with initialize(config_path='../conf'):
        cfg = compose(
            config_name='config', overrides=[f'scheduler={scheduler_name}', 'optimizer=sgd', 'private=default']
        )
        optimizer = load_obj(cfg.optimizer.class_name)(torch.nn.Linear(1, 1).parameters(), **cfg.optimizer.params)
        load_obj(cfg.scheduler.class_name)(optimizer, **cfg.scheduler.params)
