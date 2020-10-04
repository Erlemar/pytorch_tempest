# import numpy as np
# import pytest
# import torch
# from sklearn.metrics import f1_score
#
# from src.metrics.f1_score import F1Score
# from src.utils.utils import set_seed
# import subprocess
#
#
# def test_pipeline() -> None:
#     command = 'cd ..; python train.py model.encoder.params.to_one_channel=True --config-name mnist_config'
#     p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
#     out, err = p.communicate()
#     print(f'out {out} p {p}')
