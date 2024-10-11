import torch
from pytorch_forecasting.metrics import MultiHorizonMetric, MAE
import logging


class CustomMultiHorizonMetric(MultiHorizonMetric):
    """
    Custom implementation of a MultiHorizonMetric using MAE as the base loss function.

    Methods:
    --------
    loss(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        Calculates the MAE-based loss between the predictions and the target values.
    """

    def loss(self, predictions, target) -> torch.Tensor:
        """
        Computes the MAE loss between predictions and targets.

        Parameters:
        -----------
        predictions : torch.Tensor
            The predicted values (shape: (batch_size, num_horizons)).
        target : torch.Tensor
            The target values (shape: (batch_size, num_horizons)).

        Returns:
        --------
        torch.Tensor:
            The computed MAE loss.
        """
        logging.info("Calculating MAE-based loss.")
        return MAE()(predictions, target)
