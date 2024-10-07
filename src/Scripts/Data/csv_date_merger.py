import pandas as pd
import glob
import os
from functools import reduce
import logging


class CSVDateMerger:
    """
    A class to merge multiple CSV files based on a common 'date' column.

    This class reads CSV files from a specified directory, merges them
    based on a shared 'date' column, and allows for saving or returning the merged data.

    Attributes:
        folder_path (str): Path to the folder containing the CSV files.
        date_column (str): The name of the date column. Default is 'date'.
        skip_files (list): A list of filenames to skip during merging.
        merged_df (pd.DataFrame): The DataFrame containing merged data.
    """

    def __init__(self, folder_path: str, date_column: str = 'date', skip_files: list[str] = None):
        """
        Initializes a new instance of the CSVDateMerger class.

        Args:
            folder_path (str): Path to the folder containing the CSV files.
            date_column (str): The name of the date column to merge on. Default is 'date'.
            skip_files (list[str]): List of filenames to skip. Defaults to ['target_gold.csv'] if not provided.
        """
        if skip_files is None:
            skip_files = ['target_gold.csv']

        self.folder_path = folder_path
        self.date_column = date_column.lower()
        self.skip_files = skip_files
        self.merged_df = None

    def read_csv_files(self) -> list:
        """
        Reads all CSV files from the specified folder and returns a list of DataFrames.

        Returns:
            list: A list of pandas DataFrames, each representing a CSV file.
        """
        csv_files = glob.glob(os.path.join(self.folder_path, '*.csv'))
        logging.info(f'Found {len(csv_files)} CSV files in {self.folder_path}')

        dataframes = []
        for file in csv_files:
            # Extract the filename from the path to compare with skip_files
            filename = os.path.basename(file)

            # Skip the specified files
            if filename in self.skip_files:
                logging.info(f'Skipping file {filename} as specified.')
                continue

            try:
                # Read the CSV file
                df = pd.read_csv(file)
                # Convert all column names to lowercase to ensure consistency
                df.columns = df.columns.str.lower()

                # Parse the date column, ensuring it's in the right format
                if self.date_column in df.columns:
                    df[self.date_column] = pd.to_datetime(df[self.date_column])
                    logging.info(f'Read {filename} with {len(df)} rows.')
                    dataframes.append(df)
                else:
                    logging.warning(
                        f'The date column "{self.date_column}" was not found in {filename}. Skipping this file.')
            except Exception as e:
                logging.error(f'Error reading {filename}: {e}')
        return dataframes

    def merge_dataframes(self, dataframes: list) -> pd.DataFrame:
        """
        Merges a list of DataFrames on the specified date column.

        Args:
            dataframes (list): A list of pandas DataFrames to merge.

        Returns:
            pd.DataFrame: A merged DataFrame with all columns aligned by the date column.
        """
        if not dataframes:
            logging.warning('No DataFrames to merge.')
            return pd.DataFrame()

        logging.info(f'Merging {len(dataframes)} DataFrames on "{self.date_column}".')
        self.merged_df = reduce(lambda left, right: pd.merge(
            left, right, on=self.date_column, how='outer'), dataframes)

        self.merged_df.sort_values(by=self.date_column, inplace=True)
        logging.info(f'Merged DataFrame contains {len(self.merged_df)} rows and {len(self.merged_df.columns)} columns.')
        return self.merged_df

    def save_merged_csv(self, output_path: str, file_name: str = 'merged.csv'):
        """
        Saves the merged DataFrame to a specified path.

        Args:
            output_path (str): The path where the output CSV file will be saved.
            file_name (str): The name to save the output CSV file as.
        """
        if self.merged_df is not None:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                output_path = os.path.join(output_path, file_name)
                self.merged_df.to_csv(output_path, index=False)
                logging.info(f'Successfully saved merged DataFrame to {output_path}.')
            except Exception as e:
                logging.error(f'Failed to save merged DataFrame to {output_path}: {e}')
        else:
            logging.warning('Merged DataFrame is empty. Nothing to save.')

    def get_merged_dataframe(self) -> pd.DataFrame:
        """
        Returns the merged DataFrame for further use.

        Returns:
            pd.DataFrame: The merged DataFrame if it exists, otherwise an empty DataFrame.
        """
        if self.merged_df is not None:
            return self.merged_df
        else:
            logging.warning('Merged DataFrame is empty. Please run the merge process first.')
            return pd.DataFrame()

    def merge(self):
        """
        Executes the full process of reading and merging the DataFrames.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        dataframes = self.read_csv_files()
        return self.merge_dataframes(dataframes)
