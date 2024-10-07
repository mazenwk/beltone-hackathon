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
    args = parse_arguments()
    directory = DirectoryInitializer(os.path.curdir)

    input_path = args.input_path
    input_path = input_path.replace('/', os.sep).replace('\\', os.sep)
    output_path = args.output_path
    output_path = output_path.replace('/', os.sep).replace('\\', os.sep)

    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')


if __name__ == '__main__':
    main()
