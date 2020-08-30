from typing import Dict

import albumentations as A
import omegaconf
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.utils.ml_utils import stratified_group_k_fold
from src.utils.technical_utils import load_obj


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations

    Args:
        cfg:

    Returns:
        compose object
    """
    augs = []
    for a in cfg:
        if a['class_name'] == 'albumentations.OneOf':
            small_augs = []
            for small_aug in a['params']:
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v))
                    for k, v in small_aug['params'].items()
                }
                aug = load_obj(small_aug['class_name'])(**params)
                small_augs.append(aug)
            aug = load_obj(a['class_name'])(small_augs)
            augs.append(aug)

        else:
            params = {
                k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in a['params'].items()
            }
            aug = load_obj(a['class_name'])(**params)
            augs.append(aug)

    return A.Compose(augs)


def get_training_datasets(cfg: DictConfig) -> Dict:
    """
    Get datases for modelling

    Args:
        cfg: config

    Returns:
        dict with datasets
    """

    train = pd.read_csv(cfg.data.train_path)
    train = train.rename(columns={'image_id': 'image_name'})

    # for fast training
    if cfg.training.debug:
        train, valid = train_test_split(train, test_size=0.1, random_state=cfg.training.seed)
        train = train[:100]
        valid = valid[:100]

    else:

        folds = list(
            stratified_group_k_fold(X=train.index, y=train['target'], groups=train['patient_id'], k=cfg.data.n_folds)
        )
        train_idx, valid_idx = folds[cfg.data.fold_n]

        valid = train.iloc[valid_idx]
        train = train.iloc[train_idx]

    # train dataset
    dataset_class = load_obj(cfg.dataset.class_name)

    # initialize augmentations
    train_augs = load_augs(cfg['augmentation']['train']['augs'])
    valid_augs = load_augs(cfg['augmentation']['valid']['augs'])

    train_dataset = dataset_class(df=train, mode='train', img_path=cfg.data.train_image_path, transforms=train_augs)

    valid_dataset = dataset_class(df=valid, mode='valid', img_path=cfg.data.train_image_path, transforms=valid_augs)

    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset(cfg: DictConfig) -> object:
    """
    Get test dataset

    Args:
        cfg:

    Returns:
        test dataset
    """

    test_df = pd.read_csv(cfg.data.test_path)

    # valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
    # valid_augs = A.Compose(valid_augs_list)
    valid_augs = load_augs(cfg['augmentation']['valid']['augs'])
    dataset_class = load_obj(cfg.dataset.class_name)

    test_dataset = dataset_class(df=test_df, mode='test', img_path=cfg.data.test_image_path, transforms=valid_augs)

    return test_dataset
