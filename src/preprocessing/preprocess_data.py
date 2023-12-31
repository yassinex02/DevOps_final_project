"""
This module provides functionality to clean data, preprocess it, and log it to Weights & Biases.
"""
import os
import shutil
from pathlib import Path
import logging
import argparse
import pandas as pd
import wandb
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    """
    Perform data cleaning on a DataFrame.
    """
    logger.info("Starting data cleaning...")
    if df.empty:
        logger.error("Input DataFrame is empty. Cannot perform data cleaning.")
        raise ValueError("Input DataFrame is empty. Cannot perform data cleaning.")

    try:
        if drop_columns:
            drop_columns = [col for col in drop_columns if col]
            if drop_columns:
                df = df.drop(drop_columns, axis=1)
        df = df.dropna()
        df = df.drop_duplicates()
        logger.info("Data cleaning complete.")
        return df
    except Exception as e:
        logger.error("An error occurred during data cleaning: %s", str(e))
        raise


def factorize_columns(df: pd.DataFrame, columns_to_factorize: list = None) -> pd.DataFrame:
    """
    Factorize specified columns.
    """
    if not columns_to_factorize:
        logger.info("No columns specified for factorization. Returning original DataFrame.")
        return df

    try:
        df_factorized = df.copy()
        for column_name in columns_to_factorize:
            logger.info(f"Factorizing the {column_name} column.")
            df_factorized[column_name] = pd.factorize(df_factorized[column_name])[0]
        return df_factorized
    except Exception as e:
        logger.error("An error occurred while factorizing columns: %s", str(e))
        raise


def standardize_columns(df: pd.DataFrame, columns_to_standardize: list = None) -> pd.DataFrame:
    """
    Standardize specified columns using z-score normalization.
    """
    if not columns_to_standardize:
        logger.info("No columns specified for standardization. Returning original DataFrame.")
        return df

    try:
        logger.info("Standardizing columns...")
        df_standardized = df.copy()
        scaler = StandardScaler()
        df_standardized[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        return df_standardized
    except Exception as e:
        logger.error("An error occurred while standardizing columns: %s", str(e))
        raise


def main(args):
    """
    Main function to clean and preprocess data, and log to Weights & Biases.
    """
    wandb.init(job_type="data_processing")

    artifact = wandb.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    

    try:
        df = pd.read_csv(artifact_path)
        
        # Print debugging information
        print("Value of args.drop_columns before processing:", args.drop_columns)

        # Convert the string '[]' to an actual empty list
        if args.drop_columns == ['[]']:
            args.drop_columns = []

        # Print debugging information after processing
        print("Value of args.drop_columns after processing:", args.drop_columns)

        df_cleaned = clean_data(df, args.drop_columns)
        df_factorized = factorize_columns(df_cleaned, args.factorize_columns)

        print("Standardize columns Before:", args.standardize_columns)

        args.standardize_columns = args.standardize_columns[0].split(' ')

        print("Standardize columns After:", args.standardize_columns)

        df_standardized = standardize_columns(df_factorized, args.standardize_columns)

        output_artifact = wandb.Artifact(
            args.output_artifact_name,
            type=args.output_artifact_type,
            description=args.output_artifact_description
        )
        with output_artifact.new_file("processed_data.csv", mode="w") as f:
            df_standardized.to_csv(f, index=False)

        wandb.log_artifact(output_artifact)
        logger.info("Data processed and artifact logged to Weights & Biases")

    finally:
        os.remove(artifact_path)
        dir_to_remove = Path(artifact_path).parents[1]
        if dir_to_remove.exists():
            shutil.rmtree(dir_to_remove)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and log to wandb")
    parser.add_argument("--input_artifact", type=str, required=True, help="Name for the input artifact")
    parser.add_argument("--drop_columns", type=str, nargs='*', help="List of column names to drop")
    parser.add_argument("--factorize_columns", type=str, nargs='*', help="List of columns to factorize")
    parser.add_argument("--standardize_columns", type=str, nargs='*', help="List of columns to standardize")
    parser.add_argument("--output_artifact_name", type=str, required=True, help="Name for the output artifact")
    parser.add_argument("--output_artifact_type", type=str, required=True, help="Type of the output artifact")
    parser.add_argument("--output_artifact_description", type=str, help="Description for the output artifact")

    args = parser.parse_args()
    main(args)

