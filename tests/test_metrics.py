import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from src.metrics.f1_score import F1Score
from src.utils.utils import set_seed


@torch.no_grad()
@pytest.mark.parametrize('average', ['micro', 'macro', 'weighted'])
def test_f1score_metric(average: str) -> None:
    set_seed(42)
    labels = torch.randint(1, 10, (4096, 100)).flatten()
    predictions = torch.randint(1, 10, (4096, 100)).flatten()
    labels_numpy = labels.numpy()
    predictions_numpy = predictions.numpy()
    f1_metric = F1Score(average)
    my_pred = f1_metric(predictions, labels)

    f1_pred = f1_score(labels_numpy, predictions_numpy, average=average)

    assert np.isclose(my_pred.item(), f1_pred.item())
