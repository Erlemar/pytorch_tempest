import os
import random
import shutil

import numpy as np
import torch


def set_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_useful_info(new_dir: str):
    shutil.copytree(os.path.join(os.getcwd(), 'src'), os.path.join(os.getcwd(), f'{new_dir}/code/src'))
    shutil.copy2(os.path.join(os.getcwd(), 'train.py'), os.path.join(os.getcwd(), new_dir, 'code'))
