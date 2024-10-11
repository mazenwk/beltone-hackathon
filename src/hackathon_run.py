import os
import argparse
import logging
import pandas as pd
import torch
import sys

from Scripts.Initialization.directory_initializer import DirectoryInitializer
from Scripts.Logging.logging_config import configure_logging
from Scripts.Preprocess.preprocess import Preprocessor
from Scripts.module.tft_model_predictor import TFTModelPredictor
from Scripts.module.custom_metrics import CustomMultiHorizonMetric
from Scripts.Downloading_lib.downloading_libraries import download_library




def main():
    """
    Entry point for the application. Parses command-line arguments and invokes processing functions.
    """
    # Parse arguments
    args = parse_arguments()
    root_path = os.getcwd()

    # Ensure correct paths by normalizing paths and converting to absolute paths
    input_path = os.path.normpath(os.path.abspath(args.input_path))
    output_path = os.path.normpath(os.path.abspath(args.output_path))
    data_path = os.path.normpath(os.path.abspath(os.path.join(root_path, 'Data')))
    pickles_path = os.path.normpath(os.path.abspath(os.path.join(root_path, 'Pickles')))
    checkpoints_path = os.path.normpath(os.path.abspath(os.path.join(root_path, 'Checkpoints')))

    chosen_pickle = os.path.normpath(os.path.abspath(os.path.join(pickles_path, 'TFT_model.pkl')))
    chosen_checkpoint = os.path.normpath(os.path.abspath(os.path.join(checkpoints_path, 'tft_module_.pth')))

    logging.info(f'Input path set to: {input_path}')
    logging.info(f'Output path set to: {output_path}')

    
    # Download pytorch_forecasting and pytorch_lightning
    download_library('pytorch-forecasting')
    sys.path.append(os.path.abspath('Libraries'))


    # Initializes directory using the current working directory
    directory = DirectoryInitializer(root_path)
    directory.initialize()

    preprocessor = Preprocessor(input_path)
    merged = preprocessor.get_merged_dataframe()
    merged.to_csv(os.path.join(data_path, 'merged.csv'), index=True)
    merged = merged.set_index(merged['date'])

    # Initialize the model predictor with paths and other parameters
    model_predictor = TFTModelPredictor(
        input_path=input_path,
        model_path=chosen_pickle,
        checkpoint_path=chosen_checkpoint,
        num_feat_to_include=20,
        num_workers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # # Get the predictions
    predictions = model_predictor.predict(merged)

    # output df
    output_df = pd.DataFrame({'Date': merged.index[:len(predictions[0])], 'Prediction': predictions[0]})

    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Create an argument parser object
    parser = argparse.ArgumentParser(description="Process input and output file paths.")

    # Define the input and output path arguments
    parser.add_argument(
        '--input_path',
        type=str,
        default='./InputData',
        required=False,
        help='Path to the input file or directory. Defaults to "InputData/".')

    parser.add_argument(
        '--output_path',
        type=str,
        default='predictions.csv',
        required=False,  # Change to True if this argument must be provided by the user
        help='Path to the output file. Defaults to "predictions.csv".')

    return parser.parse_args()


if __name__ == '__main__':
    # Set up logging to output messages to the console & save log files
    configure_logging()
    logger = logging.getLogger(__name__)
    main()
