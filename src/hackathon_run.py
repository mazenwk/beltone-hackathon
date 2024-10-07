import os
import argparse
import logging
from Scripts.Initialization.directory_initializer import DirectoryInitializer


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


def main():
    """
    Entry point for the application. Parses command-line arguments and invokes processing functions.
    """
    # Parse arguments
    args = parse_arguments()

    # Ensure correct paths by normalizing paths and converting to absolute paths
    input_path = os.path.normpath(os.path.abspath(args.input_path))
    output_path = os.path.normpath(os.path.abspath(args.output_path))

    logging.info(f'Input path set to: {input_path}')
    logging.info(f'Output path set to: {output_path}')

    # Initializes directory using the current working directory
    directory = DirectoryInitializer(os.getcwd())
    directory.initialize()


if __name__ == '__main__':
    # Set up logging to output messages to the console
    logging.basicConfig(
        level=logging.INFO,
        format='(%(asctime)s) [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
