import os
import argparse
import logging
from datetime import datetime

from Scripts.Initialization.directory_initializer import DirectoryInitializer
from Scripts.Data.csv_date_merger import CSVDateMerger
from Scripts.Logging.logging_config import configure_logging


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

    logging.info(f'Input path set to: {input_path}')
    logging.info(f'Output path set to: {output_path}')

    # Initializes directory using the current working directory
    directory = DirectoryInitializer(root_path)
    directory.initialize()

    csv_merger = CSVDateMerger(input_path)

    # Run the process to merge the DataFrames
    data_df = csv_merger.merge()
    csv_merger.save_merged_csv(os.path.join(root_path, 'Data'))

    print(data_df.head())


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
        default='InputData/',
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
