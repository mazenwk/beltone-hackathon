import torch
import pytorch_lightning as pl
import logging
from pytorch_forecasting import TemporalFusionTransformer
from Scripts.module.custom_metrics import CustomMultiHorizonMetric

class TemporalFusionTransformerModule(pl.LightningModule):
    """
    A PyTorch Lightning module wrapper for the Temporal Fusion Transformer model.
    This module handles training, validation, and prediction steps, and integrates
    custom loss computation using a MultiHorizonMetric.

    Attributes:
    -----------
    model : TemporalFusionTransformer
        The pre-trained Temporal Fusion Transformer model.
    multi_horizon_metric : CustomMultiHorizonMetric
        The custom loss metric for evaluating multi-horizon predictions.
    """

    def __init__(self, model: TemporalFusionTransformer):
        """
        Initializes the TemporalFusionTransformerModule with a pre-trained model.

        Parameters:
        -----------
        model : TemporalFusionTransformer
            The Temporal Fusion Transformer model to be wrapped.
        """
        super(TemporalFusionTransformerModule, self).__init__()
        self.model = model
        self.multi_horizon_metric = CustomMultiHorizonMetric()
        logging.info("TemporalFusionTransformerModule initialized.")

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the custom loss for multi-horizon forecasting.

        Parameters:
        -----------
        y_hat : torch.Tensor
            The predicted values (shape: (batch_size, num_horizons, features_learned)).
        y : torch.Tensor
            The target values (shape: (batch_size, num_horizons)).

        Returns:
        --------
        torch.Tensor:
            The computed loss value.
        """
        logging.info("Computing loss for multi-horizon predictions.")
        losses = []

        # Iterate over each learned feature
        for feature_idx in range(y_hat.shape[2]):
            feature_pred = y_hat[:, :, feature_idx]  # Shape: (batch_size, num_horizons)
            feature_loss = self.multi_horizon_metric(feature_pred, y)  # Calculate loss for this feature
            losses.append(feature_loss)

        # Combine losses (e.g., mean of all feature losses)
        total_loss = torch.mean(torch.stack(losses))
        logging.info(f"Total computed loss: {total_loss.item()}")
        return total_loss

    def forward(self, x):
        """
        Performs the forward pass using the model.

        Parameters:
        -----------
        x : dict
            The input data.

        Returns:
        --------
        torch.Tensor:
            The model output.
        """
        logging.info("Performing forward pass.")
        return self.model(x)[0]

    def training_step(self, batch, batch_idx):
        """
        Defines the training step logic.

        Parameters:
        -----------
        batch : Tuple[dict, torch.Tensor]
            A batch of input data and targets.
        batch_idx : int
            The index of the batch.

        Returns:
        --------
        torch.Tensor:
            The computed training loss.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y[0])
        self.log('train_loss', loss, batch_size=y[0].size(0))
        logging.info(f"Training loss for batch {batch_idx}: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step logic.

        Parameters:
        -----------
        batch : Tuple[dict, torch.Tensor]
            A batch of input data and targets.
        batch_idx : int
            The index of the batch.

        Returns:
        --------
        None
        """
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.compute_loss(y_hat, y[0])
        self.log('val_loss', val_loss)
        logging.info(f"Validation loss for batch {batch_idx}: {val_loss.item()}")

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
        --------
        torch.optim.Optimizer:
            The configured optimizer.
        """
        logging.info("Configuring optimizer.")
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def predict_step(self, batch, batch_idx):
        """
        Defines the prediction step logic.

        Parameters:
        -----------
        batch : dict
            A batch of input data.
        batch_idx : int
            The index of the batch.

        Returns:
        --------
        torch.Tensor:
            The predicted values.
        """
        logging.info(f"Performing prediction for batch {batch_idx}.")
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
        return self(x)
