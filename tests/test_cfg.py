import os

import pytest
from hydra.experimental import compose, initialize_config_dir
from omegaconf import DictConfig

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/conf'
config_files = [f.split('.')[0] for f in os.listdir(path) if 'yaml' in f]


@pytest.mark.parametrize('config_name', config_files)
def test_cfg(config_name: str) -> None:
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/conf'
    with initialize_config_dir(config_dir=path):
        cfg = compose(config_name=config_name)
        assert isinstance(cfg, DictConfig)
