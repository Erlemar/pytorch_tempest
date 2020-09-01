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


def run(cfg: DictConfig, new_dir: str) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    cfg.callbacks.model_checkpoint.params.filepath = new_dir + cfg.callbacks.model_checkpoint.params.filepath
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

    # tb_logger = TensorBoardLogger(save_dir=cfg.general.logs_folder_name, name=cfg.general.run_dir)
    # csv_logger = CsvLogger()

    trainer = pl.Trainer(
        logger=loggers,
        early_stop_callback=EarlyStopping(**cfg.callbacks.early_stopping.params),
        checkpoint_callback=ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg)
    dm = load_obj(cfg.training.data_module_name)(hparams=hparams, cfg=cfg)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model:
        # save as a simple torch model
        model_name = cfg.general.run_dir + '/saved_models/' + cfg.general.run_dir.split('/')[-1] + '.pth'
        print(model_name)
        torch.save(model.model.state_dict(), model_name)


# @hydra.main(config_path='conf/config.yaml')
@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    # print(hydra.utils.HydraConfig.get().output_subdir, os.getcwd())
    # date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    # new_dir = f'outputs/{date}'
    os.makedirs(cfg.general.logs_dir, exist_ok=True)
    new_dir = cfg.general.run_dir
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info(new_dir)
    run(cfg, new_dir)


if __name__ == '__main__':
    run_model()
