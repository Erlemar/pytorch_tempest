# import argparse
# import glob
#
# from hydra import experimental
# import numpy as np
# import pandas as pd
# import torch
# import yaml
# from omegaconf import DictConfig, OmegaConf
#
# from src.datasets.get_dataset import get_test_dataset
# from src.utils.utils import set_seed
#
#
# def make_prediction(cfg: DictConfig) -> None:
#     """
#     Run pytorch-lightning model inference
#
#     Args:
#         cfg: hydra config
#
#     Returns:
#         None
#     """
#     set_seed(cfg.training.seed)
#     model_names = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/*')
#
#     test_dataset = get_test_dataset(cfg)
#     loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=False
#     )
#     sub = pd.read_csv(cfg.data.submission_path)
#
#     y_pred = np.zeros((len(test_dataset), len(model_names)))
#     device = cfg.data.device
#
#     for j, model_name in enumerate(model_names):
#
#         lit_model = LitMelanoma.load_from_checkpoint(checkpoint_path=model_name, cfg=cfg)
#
#         model = lit_model.model
#
#         model.to(device)
#         model.eval()
#
#         with torch.no_grad():
#
#             for ind, (img, _) in enumerate(loader):
#                 logits, _ = model(img, _)
#                 y_pred[ind * cfg.data.batch_size : (ind + 1) * cfg.data.batch_size, j] = (
#                     torch.sigmoid(logits).cpu().detach().numpy().reshape(-1)
#                 )
#
#     sub['target'] = y_pred.mean(1)
#     sub.to_csv(f'subs/{cfg.inference.run_name}_{cfg.inference.mode}.csv', index=False)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Inference in Melanoma competition')
#     parser.add_argument('--run_name', help='folder_name', type=str, default='2020_06_21_04_53_55')
#     parser.add_argument('--mode', help='valid or test', type=str, default='test')
#     args = parser.parse_args()
#
#     experimental.initialize(config_dir='conf', strict=True)
#     inference_cfg = experimental.compose(config_file='config.yaml')
#     inference_cfg['inference']['run_name'] = args.run_name
#     inference_cfg['inference']['mode'] = args.mode
#     print(inference_cfg.inference.run_name)
#     path = f'outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml'
#
#     with open(path) as cfg:
#         cfg_yaml = yaml.safe_load(cfg)
#
#     cfg_yaml['inference'] = inference_cfg['inference']
#     cfg = OmegaConf.create(cfg_yaml)
#     make_prediction(cfg)
