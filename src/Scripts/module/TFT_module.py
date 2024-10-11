import torch
from pytorch_forecasting import  TemporalFusionTransformer
import torch 
from pytorch_forecasting.metrics import MultiHorizonMetric, MAE
import pytorch_lightning as pl
class CustomMultiHorizonMetric(MultiHorizonMetric):

    def loss(self, y_pred , target) -> torch.Tensor:
        
        return MAE()(y_pred, target)
    
class TemporalFusionTransformerModule(pl.LightningModule):
    def __init__(self, model: TemporalFusionTransformer):
        super(TemporalFusionTransformerModule, self).__init__()
        self.model = model
        self.multi_horizon_metric = CustomMultiHorizonMetric()  # Initialize MultiHorizonMetric here


    def compute_loss(self, y_hat, y):
        
        # y_hat shape: (batch_size, num_horizons, features_learned)
        # y shape: (batch_size, num_horizons)

        # Initialize a list to store losses for each feature
        losses = []
        # Iterate over each learned feature
        for feature_idx in range(y_hat.shape[2]):
            feature_pred = y_hat[:, :, feature_idx]  # Shape: (batch_size, num_horizons)
            feature_loss = self.multi_horizon_metric(feature_pred, y)  # Calculate loss for this feature
            losses.append(feature_loss)

        # Combine losses (e.g., sum or average)
        total_loss = torch.mean(torch.stack(losses))  # Or use torch.sum(losses)


        return total_loss

    def forward(self, x):

        out=self.model(x)
       
        return out[0] # Ensure you're calling the model directly

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y[0])
        self.log('train_loss', loss, batch_size=y[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.compute_loss(y_hat, y[0])
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # Use model parameters
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        # Check if batch is a tuple and extract the data part
        if isinstance(batch, tuple):
            x = batch[0]  # Extract data (dictionary)
        else:
            x = batch  # If it's already a dictionary

        return self(x)  # Call the forward method with only data
