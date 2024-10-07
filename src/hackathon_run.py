import os.path
import argparse
import os

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
        required=False,  # TODO: CHANGE TO FALSE
        help='Path to the input file.')

    parser.add_argument(
        '--output_path',
        type=str,
        default='predictions.csv',
        required=False,  # TODO: CHANGE TO TRUE
        help='Path to the output file.')

    return parser.parse_args()


def main():
    """
    Entry point for the application. Parses command-line arguments and invokes processing functions.
    """
    # Parse arguments
    args = parse_arguments()
    root_path = os.getcwd()

    # Ensure correct paths
    input_path = args.input_path
    input_path = input_path.replace('/', os.sep).replace('\\', os.sep)
    input_path = os.path.join(root_path, input_path)
    output_path = args.output_path
    output_path = output_path.replace('/', os.sep).replace('\\', os.sep)
    output_path = os.path.join(root_path, output_path)

    print(f'Input path set to {input_path}')
    print(f'Output path set to {output_path}')

    # Initializes directory
    directory = DirectoryInitializer(root_path)


if __name__ == '__main__':
    main()
