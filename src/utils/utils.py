import os
import random
import shutil

import hydra
import numpy as np
import torch
import logging


def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def save_useful_info() -> None:
    logging.info(hydra.utils.get_original_cwd())
    logging.info(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), 'src'),
        os.path.join(hydra.utils.get_original_cwd(), f'{os.getcwd()}/code/src'),
    )
    shutil.copy2(
        os.path.join(hydra.utils.get_original_cwd(), 'train.py'),
        os.path.join(hydra.utils.get_original_cwd(), os.getcwd(), 'code'),
    )
