import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj, flatten_omegaconf
from src.utils.utils import set_seed, save_useful_info

warnings.filterwarnings('ignore')


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        new_dir:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    cfg.callbacks.model_checkpoint.params.filepath = os.getcwd() + cfg.callbacks.model_checkpoint.params.filepath
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
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))

    trainer = pl.Trainer(
        logger=loggers,
        # early_stop_callback=EarlyStopping(**cfg.callbacks.early_stopping.params),
        checkpoint_callback=ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg)
    dm = load_obj(cfg.datamodule.data_module_name)(hparams=hparams, cfg=cfg)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model:
        # save as a simple torch model
        # TODO save not last, but best - for this load the checkpoint and save pytorch model from it
        os.makedirs('saved_models', exist_ok=True)
        model_name = 'saved_models/best.pth'
        print(model_name)
        torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
