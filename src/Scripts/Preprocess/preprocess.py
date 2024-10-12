import logging
from typing import Tuple
import pandas as pd

from Scripts.Preprocess.data_preprocess.csv_loader import DictCSVLoader
from Scripts.Preprocess.data_preprocess.data_preprocessor import DataPreprocessor
from Scripts.Preprocess.data_preprocess.dataframe_merger import DataFrameMerger
from Scripts.Preprocess.data_preprocess.intraday_gold_processor import IntradayGoldProcessor

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from torch.utils.data import DataLoader


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

    def get_top_corr_featuers(self, merged, num_feat_to_include=20):
        """
        Gets the top features correlated with Closing Gold prices.

        Parameters:
        -----------
        merged : pd.DataFrame
            A merged DataFrame containing all the features and the 'Closing_Gold_Price' column.
        num_feat_to_include : int, optional
            The number of top features to include. Defaults to 20.

        Returns:
        -------
        list:
            A list of the top features correlated with the 'Closing_Gold_Price' column.
        """
        # `df` is the dataset with your features and gold prices
        df = merged.copy()  # Make a copy to work with
        # drop non_numeric columns
        df = df.select_dtypes(include=['int64', 'float64'])
        # Step 2: Correlation Heatmap
        correlation_matrix = df.corr()
        top_feat = correlation_matrix.nlargest(num_feat_to_include, 'closing_price').index
        features = []
        for feat in top_feat:
            features.append(feat)

        return features

    def get_timeseries_dataloader(self, merged, features, num_workers: int = 0):
        """
        Gets a TimeSeries DataLoader from a merged DataFrame containing all the features.

        Parameters:
        -----------
        merged : pd.DataFrame
            A merged DataFrame containing all the features and the 'Closing_Gold_Price' column.
        features : list
            A list of the top features correlated with the 'Closing_Gold_Price' column.
        num_workers : int, optional
            The number of workers to use in the DataLoader. Defaults to 0.

        Returns:
        -------
        DataLoader:
            A DataLoader for the TimeSeriesDataSet.
        """
        length_merged = len(merged)
        merged['time_idx'] = range(length_merged)  # Create a sequential time index
        merged['group_id'] = 0  # Use a constant group ID for all data points
        target = 'dummy_target'
        merged[target] = 0
        Normalizer = TorchNormalizer(method='robust')
        
        time_series_dataset = TimeSeriesDataSet(
            data=merged,
            scalers=Normalizer,
            time_idx='time_idx',
            allow_missing_timesteps=False,
            predict_mode=True,
            target='dummy_target',
            target_normalizer=Normalizer,
            group_ids=["group_id"],  # List of group ids; use a constant if not grouping
            min_encoder_length=60,
            max_encoder_length=100,
            min_prediction_length=1,  # Minimum prediction length (set to 1 for next time step)
            max_prediction_length=length_merged,
            static_categoricals=[],  # Include any static categorical features if available
            static_reals=[],  # Include any static real-valued features if available
            time_varying_known_categoricals=[],  # Add known categorical features if any
            time_varying_known_reals=features,  # The features that vary over time
            time_varying_unknown_categoricals=[],  # Add any unknown categorical features if available
            time_varying_unknown_reals=[]  # Add unknown real-valued features if available
        )
        dl = DataLoader(time_series_dataset, batch_size=1, shuffle=False, collate_fn=time_series_dataset._collate_fn,
                        num_workers=num_workers)
        return dl

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
            merged = closing_df.merge(data_df, on='date')
            logging.info("DataFrames merged successfully.")

            # Interpolate only numeric columns using a 3rd-degree polynomial
            numeric_columns = merged.select_dtypes(include=['number']).columns
            merged[numeric_columns] = merged[numeric_columns].interpolate(method='polynomial', order=2, limit_direction='forward')
            logging.info("Applied polynomial interpolation to numeric columns.")

            # Drop any remaining rows with NaN values
            merged.dropna(inplace=True)
            logging.info("Dropped NaN values from the merged DataFrame.")

            merged['date'] = pd.to_datetime(merged['date']).dt.date
            merged.sort_values(by='date', ascending=True, inplace=True)
            merged.reset_index(drop=True, inplace=True)

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
