import os
import pandas as pd
import logging


class DictCSVLoader:
    """
    CSVLoader class to load specified CSV files from a directory into a dictionary of pandas DataFrames.
    It supports including or excluding specific files during the loading process.
    """

    def __init__(self, input_path: str, include_list: list = None, exclude_list: list = None):
        """
        Initializes the CSVLoader with an input directory and optional include or exclude lists.

        Parameters:
        input_path (str): Path to the directory containing CSV files.
        include_list (list, optional): Default list of CSV file names (without .csv) to include.
        exclude_list (list, optional): Default list of CSV file names (without .csv) to exclude.
        """
        self.input_path = input_path
        self.default_include_list = include_list
        self.default_exclude_list = exclude_list or ['intraday_gold', 'target_gold', 'news']
        self.dataframes = {}

    def load_csv_files(self, include_list: list = None, exclude_list: list = None) -> dict:
        """
        Loads the specified CSV files into a dictionary of pandas DataFrames.

        Parameters:
        include_list (list, optional): List of CSV file names (without .csv) to include. Overrides the default.
        exclude_list (list, optional): List of CSV file names (without .csv) to exclude. Overrides the default.

        Returns:
        dict: A dictionary where keys are CSV file names (without .csv) and values are pandas DataFrames.
        """
        try:
            # Use the include/exclude lists provided in the method call or fallback to the instance's default lists
            include_list = include_list if include_list is not None else self.default_include_list
            exclude_list = exclude_list if exclude_list is not None else self.default_exclude_list

            # Get file names based on the provided include/exclude lists
            file_names = self._get_file_names(include_list, exclude_list)
            self.dataframes = self._load_files(file_names)
            logging.info(f"Successfully loaded {len(self.dataframes)} CSV files.")
        except Exception as e:
            logging.error(f"Error loading CSV files: {e}")
            raise e

        return self.dataframes

    def _get_file_names(self, include_list: list = None, exclude_list: list = None) -> list:
        """
        Retrieves the list of file names to be loaded, based on the include or exclude list.

        Parameters:
        include_list (list, optional): List of CSV file names (without .csv) to include.
        exclude_list (list, optional): List of CSV file names (without .csv) to exclude.

        Returns:
        list: A list of CSV file names (without .csv extensions).
        """
        try:
            all_files = [f for f in os.listdir(self.input_path) if f.endswith('.csv')]
            logging.info(f"Found {len(all_files)} CSV files in the directory.")

            # Remove '.csv' extension from file names
            all_files_no_ext = [os.path.splitext(f)[0] for f in all_files]

            if include_list:
                # Include only specified files
                file_names = [f for f in all_files_no_ext if f in include_list]
                logging.info(f"Including files: {file_names}")
            elif exclude_list:
                # Exclude specified files
                file_names = [f for f in all_files_no_ext if f not in exclude_list]
                logging.info(f"Excluding files: {exclude_list}, loading remaining: {file_names}")
            else:
                # Load all files if no include/exclude list is provided
                file_names = all_files_no_ext
                logging.info("No include or exclude list provided. Loading all files.")

        except Exception as e:
            logging.error(f"Error retrieving file names: {e}")
            raise e

        return file_names

    def _load_files(self, file_names: list) -> dict:
        """
        Loads the specified CSV files into a dictionary of pandas DataFrames.

        Parameters:
        file_names (list): List of CSV file names (without .csv) to be loaded.

        Returns:
        dict: A dictionary where keys are file names and values are pandas DataFrames.
        """
        dataframes = {}
        for file_name in file_names:
            try:
                file_path = os.path.join(self.input_path, f"{file_name}.csv")
                logging.info(f"Loading file: {file_path}")
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower()
                dataframes[file_name] = df
                logging.info(f"Successfully loaded {file_name}.csv")
            except FileNotFoundError:
                logging.error(f"File {file_name}.csv not found in {self.input_path}.")
            except pd.errors.EmptyDataError:
                logging.error(f"File {file_name}.csv is empty.")
            except Exception as e:
                logging.error(f"Error reading {file_name}.csv: {e}")

        return dataframes
