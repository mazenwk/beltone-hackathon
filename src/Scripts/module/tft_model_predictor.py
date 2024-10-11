import logging
from typing import Tuple
import joblib
import torch
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer

from Scripts.module.tft_module import TemporalFusionTransformerModule
from Scripts.module.custom_metrics import CustomMultiHorizonMetric
from Scripts.Preprocess.preprocess import Preprocessor


class TFTModelPredictor:
    """
    A class for loading a pre-trained Temporal Fusion Transformer (TFT) model and using it
    to make predictions on input data.

    Attributes:
    -----------
    input_path : str
        The path to the input CSVs for preprocessing.
    model_path : str
        The path to the pre-trained model file.
    checkpoint_path : str
        The path to the model checkpoint file.
    num_feat_to_include : int
        The number of top features to include in the model.
    num_workers : int
        The number of workers to use in the DataLoader.
    device : str
        The device to use for model inference ('cpu' or 'cuda').
    """

    def __init__(self, input_path: str, model_path: str, checkpoint_path: str,
                 num_feat_to_include: int = 20, num_workers: int = 0, device: str = 'cpu'):
        """
        Initializes the TFTModelPredictor with the specified parameters.

        Parameters:
        -----------
        input_path : str
            The path to the input CSVs for preprocessing.
        model_path : str
            The path to the pre-trained model file.
        checkpoint_path : str
            The path to the model checkpoint file.
        num_feat_to_include : int, optional
            The number of top features to include in the model (default is 20).
        num_workers : int, optional
            The number of workers to use in the DataLoader (default is 0).
        device : str, optional
            The device to use for model inference (default is 'cpu').
        """
        self.input_path = input_path
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.num_feat_to_include = num_feat_to_include
        self.num_workers = num_workers
        self.device = device
        logging.info(f"TFTModelPredictor initialized with input_path: {input_path}, "
                     f"model_path: {model_path}, checkpoint_path: {checkpoint_path}, "
                     f"num_feat_to_include: {num_feat_to_include}, num_workers: {num_workers}, device: {device}")

    def load_pretrained_model(self) -> TemporalFusionTransformerModule:
        """
        Loads the pre-trained TFT model from a file.

        Returns:
        --------
        TemporalFusionTransformerModule:
            The loaded and initialized TFT model.
        """
        try:
            logging.info(f"Loading pre-trained TFT model from {self.model_path}.")
            with open(self.model_path, 'rb') as file:
                loaded_model = joblib.load(file)

            model = TemporalFusionTransformerModule(loaded_model).to(self.device)
            logging.info("Model loaded successfully.")
            return model

        except FileNotFoundError as e:
            logging.error(f"File not found during model loading: {e}")
            raise e
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            raise e

    def load_model_weights(self, model: TemporalFusionTransformerModule) -> None:
        """
        Loads the weights for the pre-trained model from a checkpoint.

        Parameters:
        -----------
        model : TemporalFusionTransformerModule
            The loaded TFT model instance.
        """
        try:
            logging.info(f"Loading model weights from checkpoint: {self.checkpoint_path}.")
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            logging.info("Model weights loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found: {e}")
            raise e
        except Exception as e:
            logging.error(f"Error loading model weights: {e}")
            raise e

    def preprocess_data(self, merged: pd.DataFrame) -> Tuple[list, torch.utils.data.DataLoader]:
        """
        Preprocesses the input data and prepares the DataLoader for prediction.

        Parameters:
        -----------
        merged : pd.DataFrame
            The merged DataFrame containing all features.

        Returns:
        --------
        Tuple[list, torch.utils.data.DataLoader]:
            A tuple containing the selected feature list and the DataLoader.
        """
        try:
            logging.info("Starting data preprocessing.")
            preprocessor = Preprocessor(input_path=self.input_path)
            features = preprocessor.get_top_corr_featuers(
                merged, num_feat_to_include=self.num_feat_to_include
            )
            dataloader = preprocessor.get_timeseries_dataloader(
                merged=merged, features=features, num_workers=self.num_workers
            )
            logging.info("Data preprocessing completed successfully.")
            return features, dataloader

        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise e

    def predict(self, merged: pd.DataFrame) -> np.ndarray:
        """
        Uses the pre-trained model to make predictions based on the input data.

        Parameters:
        -----------
        merged : pd.DataFrame
            The merged DataFrame containing all features.

        Returns:
        --------
        np.ndarray:
            The predicted values as a NumPy array.
        """
        try:
            logging.info("Starting prediction process.")
            # Initialize the trainer for prediction
            trainer = Trainer()

            # Load the model and weights
            model = self.load_pretrained_model()
            self.load_model_weights(model)

            # Preprocess the data
            _, dataloader = self.preprocess_data(merged)
            print(dataloader)

            # Predict using the trained model
            predictions = trainer.predict(model, dataloaders=dataloader)
            predictions = predictions[0].mean(dim=2).numpy()
            logging.info("Prediction completed successfully.")

            return predictions

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
