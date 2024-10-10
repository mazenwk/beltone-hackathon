import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    A class to preprocess a merged DataFrame with time-series data.
    Includes steps for handling missing values, outliers, feature engineering, and scaling.

    Methods:
    --------
    convert_effr_to_numeric() -> None:
        Converts the 'effr' column to a numeric type, handling non-numeric values as NaN.

    handle_missing_values(threshold: float = 0.5, forward_fill_stocks: bool = False) -> None:
        Handles missing values by dropping columns with high percentages of missing values
        and imputing remaining missing values using forward fill for stock columns if specified.

    normalize_numerical_columns() -> None:
        Normalizes numerical columns using StandardScaler to standardize their values.

    handle_outliers() -> None:
        Detects and removes outliers using the Interquartile Range (IQR) method.

    engineer_features() -> None:
        Creates new features such as rolling averages for smoother trends.

    sort_by_date() -> None:
        Ensures the DataFrame is sorted by the 'date' column.

    preprocess(threshold: float = 0.5, forward_fill_stocks: bool = False) -> pd.DataFrame:
        Executes all preprocessing steps and returns the processed DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the DataPreprocessor with a DataFrame.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the merged data to preprocess.
        """
        self.df = dataframe

    def convert_effr_to_numeric(self) -> None:
        """
        Converts the 'effr' column to a numeric type, handling non-numeric values as NaN.
        """
        try:
            logging.info("Converting 'effr' column to numeric.")
            self.df['effr'] = pd.to_numeric(self.df['effr'], errors='coerce')
            logging.info("Successfully converted 'effr' column to numeric.")
        except Exception as e:
            logging.error(f"Error converting 'effr' column to numeric: {e}")
            raise e

    def handle_missing_values(self, threshold: float = 0.5, forward_fill_stocks: bool = False) -> None:
        """
        Handles missing values by dropping columns with a high percentage of missing data
        and imputing remaining missing values using forward fill for stock columns if specified.

        Parameters:
        -----------
        threshold : float, optional
            The threshold for dropping columns (default is 0.5). Columns with more than
            the specified percentage of missing values will be dropped.
        forward_fill_stocks : bool, optional
            If True, forward fill is applied to stock-related columns; if False, skips forward fill
            for stock columns and only applies it to other columns.
        """
        try:
            logging.info(f"Handling missing values with threshold: {threshold}")

            # Drop columns with missing values above the threshold
            cols_to_drop = self.df.columns[self.df.isnull().mean() > threshold]
            self.df.drop(columns=cols_to_drop, inplace=True)
            logging.info(f"Dropped columns with high missing values: {list(cols_to_drop)}")

            # Define stock columns based on common patterns in the column names
            stock_columns = [col for col in self.df.columns if 'stock' in col.lower()]

            if forward_fill_stocks:
                # Apply forward fill to all columns
                logging.info("Applying forward fill to all columns, including stock-related columns.")
                self.df.ffill(inplace=True)
            else:
                # Apply forward fill to non-stock columns only
                logging.info("Applying forward fill to non-stock columns only.")
                non_stock_columns = [col for col in self.df.columns if col not in stock_columns]
                self.df.loc[:, non_stock_columns].ffill(inplace=True)

            logging.info("Imputed remaining missing values using forward fill.")
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")
            raise e

    def normalize_numerical_columns(self) -> None:
        """
        Normalizes numerical columns using StandardScaler to standardize their values.
        """
        try:
            logging.info("Normalizing numerical columns.")
            numeric_cols = self.df.select_dtypes(include=['float64']).columns
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            logging.info("Successfully normalized numerical columns.")
        except Exception as e:
            logging.error(f"Error normalizing numerical columns: {e}")
            raise e

    def handle_outliers(self) -> None:
        """
        Detects and removes outliers using the Interquartile Range (IQR) method.
        """
        try:
            logging.info("Handling outliers using the IQR method.")
            numeric_cols = self.df.select_dtypes(include=['float64']).columns
            Q1 = self.df[numeric_cols].quantile(0.25)
            Q3 = self.df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            condition = ~((self.df[numeric_cols] < (Q1 - 1.5 * IQR)) | (self.df[numeric_cols] > (Q3 + 1.5 * IQR)))
            self.df = self.df[condition.all(axis=1)]
            logging.info("Outliers removed successfully.")
        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise e

    def engineer_features(self) -> None:
        """
        Creates new features such as rolling averages for smoother trends.
        """
        try:
            logging.info("Engineering features (e.g., rolling averages).")
            pd.options.mode.copy_on_write = True
            self.df.loc[:, 'wti_7_day_avg'] = self.df['wti oil price fob (dollars per barrel)'].rolling(window=7).mean()
            self.df.loc[:, 'wti_30_day_avg'] = self.df['wti oil price fob (dollars per barrel)'].rolling(
                window=30).mean()
            logging.info("Feature engineering completed successfully.")
        except KeyError as ke:
            logging.error(f"KeyError during feature engineering: {ke}")
            raise ke
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}")
            raise e

    def sort_by_date(self) -> None:
        """
        Ensures the DataFrame is sorted by the 'date' column.
        """
        try:
            logging.info("Sorting DataFrame by 'date' column.")
            self.df.sort_values(by='date', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logging.info("DataFrame sorted by 'date' column.")
        except KeyError:
            logging.error("The 'date' column is missing from the DataFrame.")
            raise KeyError("The 'date' column is required but missing.")
        except Exception as e:
            logging.error(f"Error sorting by 'date' column: {e}")
            raise e

    def preprocess(self, threshold: float = 0.5, forward_fill_stocks: bool = False,
                   handle_outliers: bool = True) -> pd.DataFrame:
        """
        Executes all preprocessing steps and returns the processed DataFrame.

        Parameters:
        -----------
        threshold : float, optional
            The threshold for dropping columns based on missing values (default is 0.5).
        forward_fill_stocks : bool, optional
            If True, forward fill is applied to stock-related columns; if False, skips forward fill
            for stock columns and only applies it to other columns.

        Returns:
        --------
        pd.DataFrame:
            The preprocessed DataFrame.
        """
        try:
            logging.info("Starting preprocessing steps.")
            self.convert_effr_to_numeric()
            self.handle_missing_values(threshold, forward_fill_stocks)
            self.normalize_numerical_columns()
            if handle_outliers:
                self.handle_outliers()
            self.engineer_features()
            self.sort_by_date()
            logging.info("Preprocessing completed successfully.")
            return self.df
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e
