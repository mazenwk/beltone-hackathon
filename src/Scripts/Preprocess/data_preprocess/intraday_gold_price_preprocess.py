import logging
import pandas as pd


class IntradayGoldPricesPreprocessor:
    """
    Preprocesses intraday gold prices to extract the closing price for each day.

    This class takes either a pandas DataFrame or a path to a CSV file containing
    intraday gold prices, validates the input, and provides functionality to preprocess
    the data, including converting the 'timestamp' column to UTC dates.

    Attributes:
        intraday_prices_df (pd.DataFrame): DataFrame containing the intraday gold prices.
        closing_prices (pd.DataFrame): DataFrame containing the daily closing prices.
    """

    def __init__(self, intraday_prices_df: pd.DataFrame = None, intraday_prices_csv_path: str = None):
        """
        Initializes the IntradayGoldPricesPreprocessor with a DataFrame or a CSV path.

        Args:
            intraday_prices_df (pd.DataFrame, optional): DataFrame containing intraday prices.
            intraday_prices_csv_path (str, optional): Path to a CSV file containing intraday prices.
                Used if the DataFrame is empty or not provided.

        Raises:
            ValueError: If both the DataFrame is empty and no valid CSV path is provided.
        """
        self.intraday_prices_df = pd.DataFrame()
        self.closing_prices = pd.DataFrame()
        self._validate_args(intraday_prices_df, intraday_prices_csv_path)

    def _validate_args(self, intraday_prices_df: pd.DataFrame = None, intraday_prices_csv_path: str = None):
        """
        Validates the provided DataFrame and loads data from CSV if necessary.

        If the DataFrame is not empty, it is assigned to the instance attribute.
        If the DataFrame is empty, attempts to read data from the provided CSV path.

        Args:
            intraday_prices_df (pd.DataFrame, optional): DataFrame containing intraday prices.
            intraday_prices_csv_path (str, optional): Path to a CSV file with intraday prices.

        Raises:
            ValueError: If both the DataFrame is empty and no valid CSV path is provided.
        """
        if intraday_prices_df is not None and not intraday_prices_df.empty:
            logging.info("Using provided DataFrame for intraday prices.")
            self.intraday_prices_df = intraday_prices_df
        elif intraday_prices_csv_path:
            logging.info(f'Loading intraday prices from CSV at: {intraday_prices_csv_path}')
            try:
                self.intraday_prices_df = pd.read_csv(intraday_prices_csv_path)
                logging.info(f'Successfully loaded data from {intraday_prices_csv_path}.')
            except Exception as e:
                logging.error(f'Error loading CSV: {e}')
                raise ValueError(f"Failed to read CSV from {intraday_prices_csv_path}") from e
        else:
            raise ValueError("A non-empty DataFrame or a valid CSV path must be provided.")

        # Standardize column names to lowercase for consistency
        self.intraday_prices_df.columns = self.intraday_prices_df.columns.str.lower()

    def preprocess(self) -> pd.DataFrame:
        """
        Orchestrates the preprocessing of intraday gold prices data.

        This method converts the 'timestamp' column to UTC datetime format,
        extracts only the date part, and calculates the closing price for each day.

        Returns:
            pd.DataFrame: A DataFrame containing the daily closing prices with columns 'date' and 'closing_price'.
        """
        if not self._convert_timestamps():
            logging.error("'Timestamp' column not found. Aborting preprocessing.")
            return self.intraday_prices_df

        # Extract and store the closing prices
        self.closing_prices = self._extract_closing_prices()
        return self.closing_prices

    def _extract_closing_prices(self) -> pd.DataFrame:
        """
        Extracts the closing price for each date from the intraday data.

        This method sorts the intraday data by 'timestamp', groups it by 'date',
        and retrieves the last recorded price for each day as the closing price.

        Returns:
            pd.DataFrame: A DataFrame containing 'date' and 'closing_price'.
        """
        logging.info("Extracting daily closing prices.")
        # Sort and group by 'date' to get the last price of each day.
        closing_prices_df = (
            self.intraday_prices_df
            .sort_values(by='timestamp')
            .groupby('date', as_index=False)
            .last()
            .rename(columns={'24k': 'closing_price'})[['date', 'closing_price']]
        )

        logging.info("Daily closing prices extracted successfully.")
        return closing_prices_df

    def _convert_timestamps(self) -> bool:
        """
        Converts the 'timestamp' column to a UTC datetime object and extracts the date part.

        Uses flexible parsing to handle mixed date formats in the 'timestamp' column.

        Returns:
            bool: True if the conversion was successful, False if the 'timestamp' column is missing.
        """
        if 'timestamp' not in self.intraday_prices_df.columns:
            logging.error("'Timestamp' column not found.")
            return False

        try:
            # Use flexible parsing with 'mixed' to handle varying formats, and convert to UTC
            self.intraday_prices_df['timestamp'] = pd.to_datetime(
                self.intraday_prices_df['timestamp'], format='mixed', utc=True, errors='coerce'
            )
            # Drop rows where parsing failed, if any
            self.intraday_prices_df.dropna(subset=['timestamp'], inplace=True)
            # Extract the date from the parsed timestamps
            self.intraday_prices_df['date'] = self.intraday_prices_df['timestamp'].dt.date
            logging.info("Timestamps converted to UTC and dates extracted.")
            return True
        except Exception as e:
            logging.error(f"Error converting timestamps: {e}")
            return False
