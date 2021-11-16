import torch
from torch import nn


class VentilatorMAE(nn.Module):
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self):
        """
        Init.

        Args:
            average: averaging method
        """
        super().__init__()

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor, u_out: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        weights = 1 - u_out
        mae = (labels - predictions).abs() * weights
        return mae.sum(-1) / weights.sum(-1)
