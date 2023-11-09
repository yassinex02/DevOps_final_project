import argparse
import logging
from typing import List
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Load configurations from config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(
    format=config["logging"]["format"], level=config["logging"]["level"].upper()
)


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
        logging.error("Input DataFrame is empty. Cannot perform data cleaning.")
        raise ValueError("Input DataFrame is empty. Cannot perform data cleaning.")

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
            f"An error occurred while factorizing the {column_name} column: {e}"
        )
        raise


def standardize_columns(
    df: pd.DataFrame, columns_to_standardize: List[str]
) -> pd.DataFrame:
    """
    Standardize specified columns using z-score normalization.
    """
    try:
        logging.info(f"Standardizing columns: {columns_to_standardize}")
        df_standardized = df.copy()
        scaler = StandardScaler()
        df_standardized[columns_to_standardize] = scaler.fit_transform(
            df[columns_to_standardize]
        )
        return df_standardized
    except Exception as e:
        logging.error(f"An error occurred while standardizing columns: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "output_file", type=str, help="Path to save the cleaned data CSV file"
    )
    parser.add_argument("--drop_columns", nargs="+", help="List of columns to drop")

    args = parser.parse_args()

    # Read data from CSV
    df = pd.read_csv(args.input_file)

    # Clean data
    if args.drop_columns:
        df = clean_data(df, drop_columns=args.drop_columns)

    # Factorize columns from config.yaml
    factorize_columns = config.get("factorize_columns", [])
    for column in factorize_columns:
        df = factorize_column(df, column_name=column)

    # Standardize columns from config.yaml
    standardize_columns_list = config.get("standardize_columns", [])
    if standardize_columns_list:
        df = standardize_columns(df, columns_to_standardize=standardize_columns_list)

    # Save cleaned data
    df.to_csv(args.output_file, index=False)
    logging.info(f"Cleaned data saved to {args.output_file}")


if __name__ == "__main__":
    main()
