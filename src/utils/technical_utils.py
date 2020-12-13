import collections
import importlib
from itertools import product
from typing import Any, Dict, Generator

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def convert_to_jit(model: nn.Module, save_name: str, cfg: DictConfig) -> None:
    input_shape = (1, 3, cfg.datamodule.main_image_size, cfg.datamodule.main_image_size)
    input_shape1 = 1
    out_path = f'saved_models/{save_name}_jit.pt'
    model.eval()

    device = next(model.parameters()).device
    input_tensor = torch.ones(input_shape).float().to(device)
    input_tensor1 = torch.ones(input_shape1, dtype=torch.long).to(device)
    traced_model = torch.jit.trace(model, (input_tensor, input_tensor1))
    torch.jit.save(traced_model, out_path)


def product_dict(**kwargs: Dict) -> Generator:
    """
    Convert dict with lists in values into lists of all combinations

    This is necessary to convert config with experiment values
    into format usable by hydra
    Args:
        **kwargs:

    Returns:
        list of lists

    ---
    Example:
        >>> list_dict = {'a': [1, 2], 'b': [2, 3]}
        >>> list(product_dict(**list_dict))
        >>> [['a=1', 'b=2'], ['a=1', 'b=3'], ['a=2', 'b=2'], ['a=2', 'b=3']]

    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        zip_list = list(zip(keys, instance))
        yield [f'{i}={j}' for i, j in zip_list]


def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of parameter

    This fuction is used to get key names which can be used in hydra.

    Args:
        cfg:

    Returns:
        converted dict

    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f'{k}.{k1}'] = v1

    return experiment_dict


def flatten_omegaconf(d, sep='_'):
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=''):

        if isinstance(t, list):
            for i, _ in enumerate(t):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if isinstance(v, (int, float))}
    # obj = {k: v for k, v in obj.items()}

    return obj
