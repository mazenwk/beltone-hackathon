import logging
from typing import Tuple
import pandas as pd

from src.Scripts.Preprocess.data_preprocess.csv_loader import DictCSVLoader
from src.Scripts.Preprocess.data_preprocess.data_preprocessor import DataPreprocessor
from src.Scripts.Preprocess.data_preprocess.dataframe_merger import DataFrameMerger
from src.Scripts.Preprocess.data_preprocess.intraday_gold_processor import IntradayGoldProcessor


class Preprocessor:
    """
    A class to handle the loading, preprocessing, and merging of various data sources.
    It includes methods for handling intraday gold prices, data preprocessing, and merging
    dataframes, ensuring a ready-to-use DataFrame for modeling.

    Attributes:
    -----------
    input_path : str
        The path to the directory containing input CSV files.
    loader : DictCSVLoader
        An instance of DictCSVLoader to load CSV files as DataFrames.
    """

    def __init__(self, input_path: str):
        """
        Initializes the Preprocessor with a specified input path.

        Parameters:
        -----------
        input_path : str
            The path to the directory containing input CSV files.
        """
        self.input_path = input_path
        self.loader = DictCSVLoader(self.input_path)
        logging.info(f"Preprocessor initialized with input path: {self.input_path}")

    def get_merged_dataframe(self) -> pd.DataFrame:
        """
        Preprocesses data and intraday gold prices, merges them, and applies polynomial interpolation.

        Returns:
        --------
        pd.DataFrame
            The merged DataFrame after applying preprocessing and interpolation.
        """
        try:
            logging.info("Starting the process to get the merged DataFrame.")

            # Get processed data and closing prices DataFrames
            data_df = self.get_processed_data_df(threshold=0.5, forward_fill_stocks=False, handle_outliers=False)
            closing_df = self.get_processed_closing_prices_df()

            # Merge the DataFrames on the 'date' column
            merged = data_df.merge(closing_df, on='date', how='left')
            logging.info("DataFrames merged successfully.")

            # Interpolate only numeric columns using a 3rd-degree polynomial
            numeric_columns = merged.select_dtypes(include=['number']).columns
            merged[numeric_columns] = merged[numeric_columns].interpolate(method='polynomial', order=2)
            logging.info("Applied polynomial interpolation to numeric columns.")

            # Drop any remaining rows with NaN values
            merged.dropna(inplace=True)
            logging.info("Dropped NaN values from the merged DataFrame.")

            return merged

        except Exception as e:
            logging.error(f"Error in get_merged_dataframe: {e}")
            raise e

    def get_processed_data_df(self, threshold: float = 0.1, forward_fill_stocks: bool = False,
                              handle_outliers: bool = True) -> pd.DataFrame:
        """
        Loads and preprocesses the main data DataFrame.

        Parameters:
        -----------
        threshold : float, optional
            The threshold for dropping columns based on missing values (default is 0.1).
        forward_fill_stocks : bool, optional
            If True, forward fill is applied to stock-related columns.
        handle_outliers : bool, optional
            If True, handles outliers in the DataFrame.

        Returns:
        --------
        pd.DataFrame
            The preprocessed main data DataFrame.
        """
        try:
            logging.info("Starting to process the main data DataFrame.")
            data_df = self._load_merged_data_df()
            data_preprocessor = DataPreprocessor(data_df)
            processed_df = data_preprocessor.preprocess(
                threshold=threshold, forward_fill_stocks=forward_fill_stocks, handle_outliers=handle_outliers
            )
            logging.info("Data preprocessing completed successfully.")
            return processed_df

        except Exception as e:
            logging.error(f"Error in get_processed_data_df: {e}")
            raise e

    def get_processed_closing_prices_df(self) -> pd.DataFrame:
        """
        Processes intraday gold price data to obtain daily closing prices.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing daily closing prices of gold.
        """
        try:
            logging.info("Starting to process intraday gold prices for closing prices.")
            _, intraday_dict, _, _ = self._load_dataframes_dict_from_csvs(load_intraday=True)
            intraday_preprocessor = IntradayGoldProcessor(intraday_dict)
            closing_prices_df = intraday_preprocessor.process_intraday_data()
            logging.info("Intraday gold prices processed successfully.")
            return closing_prices_df

        except Exception as e:
            logging.error(f"Error in get_processed_closing_prices_df: {e}")
            raise e

    def _load_merged_data_df(self) -> pd.DataFrame:
        """
        Loads and merges DataFrames for the main data.

        Returns:
        --------
        pd.DataFrame
            The merged DataFrame of the main data.
        """
        try:
            logging.info("Loading and merging main data DataFrames.")
            data_dict, _, _, _ = self._load_dataframes_dict_from_csvs(load_data=True)
            merger = DataFrameMerger(data_dict)
            merger.format_date_columns()
            data_df = merger.merge_dataframes()
            logging.info("Main data DataFrames loaded and merged successfully.")
            return data_df

        except Exception as e:
            logging.error(f"Error in _load_merged_data_df: {e}")
            raise e

    def _load_dataframes_dict_from_csvs(
            self, load_data: bool = False, load_intraday: bool = False,
            load_target: bool = False, load_news: bool = False) -> Tuple[dict, dict, dict, dict]:
        """
        Loads various CSV files into DataFrame dictionaries.

        Parameters:
        -----------
        load_data : bool, optional
            If True, loads the main data CSV files.
        load_intraday : bool, optional
            If True, loads the intraday gold price data.
        load_target : bool, optional
            If True, loads the target CSV files.
        load_news : bool, optional
            If True, loads the news data CSV files.

        Returns:
        --------
        Tuple[dict, dict, dict, dict]
            A tuple containing dictionaries of DataFrames for data, intraday, target, and news.
        """
        try:
            logging.info("Loading CSV files into DataFrame dictionaries.")
            data_dict = intraday_dict = target_dict = news_dict = None

            if load_data:
                data_dict = self.loader.load_csv_files()
                logging.info("Loaded main data CSV files.")
            if load_intraday:
                intraday_dict = self.loader.load_csv_files(include_list=['intraday_gold'])
                logging.info("Loaded intraday gold price data.")
            if load_target:
                target_dict = self.loader.load_csv_files(include_list=['target_gold'])
                logging.info("Loaded target gold data.")
            if load_news:
                news_dict = self.loader.load_csv_files(include_list=['news'])
                logging.info("Loaded news data.")

            return data_dict, intraday_dict, target_dict, news_dict

        except Exception as e:
            logging.error(f"Error in _load_dataframes_dict_from_csvs: {e}")
            raise e
