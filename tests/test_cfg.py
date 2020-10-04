import os

from hydra.experimental import compose, initialize_config_dir
from omegaconf import DictConfig


def test_cfg() -> None:
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/conf'
    with initialize_config_dir(config_dir=path):
        cfg = compose(config_name='config')
        assert isinstance(cfg, DictConfig)
