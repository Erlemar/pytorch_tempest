import logging
import os
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj, convert_to_jit
from src.utils.utils import set_seed, save_useful_info

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model


    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    run_name = os.path.basename(os.getcwd())
    cfg.callbacks.model_checkpoint.params.dirpath = Path(
        os.getcwd(), cfg.callbacks.model_checkpoint.params.dirpath
    ).as_posix()
    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
            elif 'name' in logger.params.keys():
                logger.params['name'] = run_name

            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))
    callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint.params))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightning_module_name)(cfg=cfg)
    dm = load_obj(cfg.datamodule.data_module_name)(cfg=cfg)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model and cfg.general.save_best:
        if os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder
            save_name = os.path.basename(os.path.normpath(best_path))
            logging.info(f'{save_name = }')
            model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(best_path, cfg=cfg, strict=False)
            model_name = Path(
                cfg.callbacks.model_checkpoint.params.dirpath, f'best_{save_name}'.replace('.ckpt', '.pth')
            ).as_posix()
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)

    if cfg.general.convert_to_jit and os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
        best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
        save_name = os.path.basename(os.path.normpath(best_path))
        convert_to_jit(model, save_name, cfg)

    logging.info(f'{run_name = }')


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    logging.info(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
