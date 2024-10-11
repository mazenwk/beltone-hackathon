import joblib
import torch
import os
from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
import pandas as pd
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_lightning import Trainer
from Scripts.module.TFT_module import TemporalFusionTransformerModule, CustomMultiHorizonMetric
from Scripts.Preprocess.preprocess import Preprocessor

def TFT_model(input_path, merged: pd.DataFrame, num_feat_to_include: int = 20, num_workers: int=0, device='cpu'):
    """
    Load a pre-trained Temple Fusion Transformer model from a pickle file and use it to make predictions on input data.

    Parameters:
    input_path (str): The input_path for the CSVs.
    merged (pd.DataFrame): The merged DataFrame containing all features.
    num_feat_to_include (int): The number of top features to include in the model (default is 20).
    num_workers (int): The number of workers to use in the DataLoader (default is 0).
    device (str): The device to use for training (default is 'cpu').

    Returns:
    np.ndarray: The predicted values based on the input features.
    """
    # Init the trainer to predict from the model
    trainer = Trainer()
    # Load the pre-trained TFT model 
    with open('src/Pickles/TFT_model.pkl', 'rb') as file:
        loaded_model = joblib.load(file)

    loaded_module = TemporalFusionTransformerModule(loaded_model).to(device)
    loaded_module.load_state_dict(torch.load(os.path.join('src/Checkpoints/', 'tft_module_.pth'), weights_only=True))
    # Data preprocessing for the input data
    My_preprocessor = Preprocessor(input_path=input_path)
    features = My_preprocessor.get_top_features_corr_with_Closing_Gold_prices(merged, num_feat_to_include=num_feat_to_include)

    dataloader = My_preprocessor.get_TimeSeries_dataloader(merged=merged, features=features, num_workers=num_workers)

    # Predict based on the input features
    y_pred = trainer.predict(loaded_module, dataloaders=dataloader)
    y_pred = y_pred[0].mean(dim=2).numpy()

    return y_pred
