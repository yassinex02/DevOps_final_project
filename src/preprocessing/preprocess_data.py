import pandas as pd
import logging
import yaml
from typing import List
from sklearn.preprocessing import StandardScaler


# Load configurations from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(
    format=config['logging']['format'], level=config['logging']['level'].upper())


def clean_data(df: pd.DataFrame, drop_columns: List[str] = None) -> pd.DataFrame:
    """
    Perform data cleaning on a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        drop_columns (List[str], optional): List of column names to drop from the DataFrame. Default is None.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    logging.info("Starting data cleaning...")

    if df.empty:
        logging.error(
            "Input DataFrame is empty. Cannot perform data cleaning.")
        raise ValueError(
            "Input DataFrame is empty. Cannot perform data cleaning.")

    try:
        # Initialize df_cleaned
        df_cleaned = df

        # Drop specified columns
        if drop_columns:
            df_cleaned = df.drop(drop_columns, axis=1)

        # Drop NA values
        df_cleaned = df_cleaned.dropna()

        # Drop duplicates
        df_cleaned = df_cleaned.drop_duplicates()

        logging.info("Data cleaning complete.")

        return df_cleaned

    except Exception as e:
        logging.error(f"An error occurred during data cleaning: {e}")
        raise


def factorize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Factorize the specified column.
    """
    try:
        logging.info(f"Factorizing the {column_name} column.")
        df_factorized = df.copy()
        df_factorized[column_name] = pd.factorize(df[column_name])[0]
        return df_factorized
    except Exception as e:
        logging.error(
            f"An error occurred while factorizing the {column_name} column: {e}")
        raise


def standardize_columns(df: pd.DataFrame, columns_to_standardize: List[str]) -> pd.DataFrame:
    """
    Standardize specified columns using z-score normalization.
    """
    try:
        logging.info(f"Standardizing columns: {columns_to_standardize}")
        df_standardized = df.copy()
        scaler = StandardScaler()
        df_standardized[columns_to_standardize] = scaler.fit_transform(
            df[columns_to_standardize])
        return df_standardized
    except Exception as e:
        logging.error(f"An error occurred while standardizing columns: {e}")
        raise
