import pandas as pd
import logging


class IntradayGoldProcessor:
    """
    A class to process intraday gold price data, compute daily closing prices, and generate lag features.
    It extracts relevant information, processes timestamps, and returns a DataFrame ready for further analysis.

    Methods:
    --------
    process_intraday_data(intraday_dict: dict) -> pd.DataFrame:
        Extracts and processes the intraday gold price data from a dictionary and returns a DataFrame
        with date, closing prices, and lag features.
    """

    def __init__(self, intraday_dict: dict):
        """
        Initializes the IntradayGoldProcessor with a dictionary containing intraday gold price data.

        Parameters:
        -----------
        intraday_dict : dict
            A dictionary containing a single DataFrame with intraday gold prices.
        """
        if not intraday_dict or len(intraday_dict) != 1:
            raise ValueError("The intraday_dict must contain exactly one DataFrame.")

        self.intraday_df = next(iter(intraday_dict.values()))

    def process_intraday_data(self) -> pd.DataFrame:
        """
        Processes the intraday gold price data and computes daily closing prices with lag features.

        Returns:
        --------
        pd.DataFrame:
            A DataFrame with columns 'date', 'closing_price', 'lag1', and 'lag2'.
        """
        try:
            logging.info("Processing intraday gold price data.")

            # Convert the 'timestamp' column to datetime while preserving all entries
            # self.intraday_df['timestamp'] = pd.to_datetime(
            #     self.intraday_df['timestamp'], utc=True, errors='coerce'
            # )
            # self.intraday_df['date'] = self.intraday_df['timestamp'].dt.normalize()
            self.intraday_df['date'] = pd.to_datetime(self.intraday_df['timestamp'], format='mixed', utc=True).dt.date
            self.intraday_df['date'] = pd.to_datetime(self.intraday_df['date'], utc=True)
            logging.info("Converted 'timestamp' to datetime and extracted 'date' column.")

            # Calculate the closing price for each date
            closing_prices_df = self.intraday_df.groupby('date')['24k'].last().reset_index()
            closing_prices_df.rename(columns={'24k': 'closing_price'}, inplace=True)
            logging.info("Calculated daily closing prices.")

            # Create lag features for the closing prices
            closing_prices_df['lag1'] = closing_prices_df['closing_price'].shift(1)
            closing_prices_df['lag2'] = closing_prices_df['closing_price'].shift(2)
            logging.info("Generated lag features 'lag1' and 'lag2'.")

            # Handle missing or invalid values and drop NaN rows
            closing_prices_df.dropna(inplace=True)
            closing_prices_df = closing_prices_df[closing_prices_df['closing_price'] > 0]
            logging.info("Cleaned data and removed NaN or invalid values.")

            return closing_prices_df

        except Exception as e:
            logging.error(f"Error processing intraday gold data: {e}")
            raise e
