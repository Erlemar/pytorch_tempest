import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj
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
    run_name = os.path.basename(os.getcwd())
    hparams = OmegaConf.to_container(cfg)

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
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))

    trainer = pl.Trainer(
        logger=loggers,
        checkpoint_callback=ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        callbacks=callbacks,
        **cfg.trainer,
    )

    dm = load_obj(cfg.datamodule.data_module_name)(hparams=hparams, cfg=cfg)
    dm.setup()
    model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg, tag_to_idx=dm.tag_to_idx)
    model._vectorizer = dm._vectorizer
    # dm = load_obj(cfg.datamodule.data_module_name)(hparams=hparams, cfg=cfg)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model:
        if cfg.general.save_best:
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder and extension
            save_name = best_path.split('/')[-1][:-5]
            model = model.load_from_checkpoint(
                best_path, hparams=hparams, cfg=cfg, tag_to_idx=dm.tag_to_idx, strict=False
            )
            model_name = f'saved_models/{save_name}.pth'
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path='conf', config_name='config_ner')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
