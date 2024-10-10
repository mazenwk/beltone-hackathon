import pandas as pd
import logging


class DataFrameMerger:
    """
    A class to format, update, and merge multiple DataFrames based on their date columns.
    Ensures consistent date formatting and merges all DataFrames into a single DataFrame.

    Attributes:
    -----------
    dataframes : dict
        A dictionary of pandas DataFrames where keys are identifiers and values are DataFrames.

    Methods:
    --------
    format_date_columns(date_format: str = "%Y-%m-%d", utc: bool = True) -> None:
        Formats the 'date' column in each DataFrame to a consistent format and updates the column type to datetime.

    merge_dataframes() -> pd.DataFrame:
        Merges all DataFrames on the 'date' column and returns the merged DataFrame sorted from older to newer dates.
    """

    def __init__(self, dataframes: dict):
        """
        Initializes the DataFrameMerger with a dictionary of DataFrames.

        Parameters:
        -----------
        dataframes : dict
            A dictionary where keys are DataFrame identifiers and values are pandas DataFrames.
        """
        self.dataframes = dataframes

    def format_date_columns(self, date_format: str = "%Y-%m-%d", utc: bool = True) -> None:
        """
        Formats the 'date' column in each DataFrame to a consistent format and updates the column type to datetime.

        Parameters:
        -----------
        date_format : str, optional
            The date format to which all 'date' columns should be converted (default is '%Y-%m-%d').
        utc : bool, optional
            If True, converts the dates to UTC time (default is True).

        Raises:
        -------
        ValueError:
            If a DataFrame does not contain a 'date' column.
        """
        for key, df in self.dataframes.items():
            try:
                logging.info(f"Formatting 'date' column in DataFrame: {key}.")

                if 'date' not in df.columns:
                    raise ValueError(f"DataFrame '{key}' does not contain a 'date' column.")

                # Convert the 'date' column to datetime and format it
                df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=utc)
                df['date'] = df['date'].dt.strftime(date_format)

                # Re-convert formatted string dates back to datetime
                df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=utc)

                self.dataframes[key] = df
                logging.info(f"'date' column formatted successfully in DataFrame: {key}.")
            except ValueError as ve:
                logging.error(f"ValueError encountered for DataFrame '{key}': {ve}")
                raise ve
            except Exception as e:
                logging.error(f"Error while formatting 'date' column in DataFrame '{key}': {e}")
                raise e

    def merge_dataframes(self) -> pd.DataFrame:
        """
        Merges all DataFrames on the 'date' column and sorts the merged DataFrame by date.

        Returns:
        --------
        pd.DataFrame:
            A merged DataFrame containing all input DataFrames, aligned by 'date' and sorted chronologically.

        Raises:
        -------
        ValueError:
            If no DataFrames are available for merging.
        """
        try:
            if not self.dataframes:
                raise ValueError("No DataFrames available for merging.")

            logging.info("Starting to merge DataFrames on 'date' column.")

            # Initialize the merged DataFrame with the first DataFrame in the dictionary
            merged_df = None
            for key, df in self.dataframes.items():
                if merged_df is None:
                    merged_df = df
                else:
                    # Merge the DataFrames on 'date' using an outer join to include all dates
                    merged_df = pd.merge(merged_df, df, on='date', how='outer', suffixes=('', f'_{key}'))

                logging.info(f"Merged DataFrame '{key}' into the consolidated DataFrame.")

            # Sort the final merged DataFrame by the 'date' column from older to newer
            merged_df.sort_values(by='date', inplace=True)
            merged_df.reset_index(drop=True, inplace=True)

            logging.info("DataFrames merged successfully.")
            return merged_df

        except ValueError as ve:
            logging.error(f"ValueError during merging: {ve}")
            raise ve
        except Exception as e:
            logging.error(f"Error during DataFrame merging: {e}")
            raise e
