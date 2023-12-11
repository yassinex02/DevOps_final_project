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


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    """
    Perform data cleaning on a DataFrame.
    """
    logger.info("Starting data cleaning...")
    if df.empty:
        logger.error("Input DataFrame is empty. Cannot perform data cleaning.")
        raise ValueError(
            "Input DataFrame is empty. Cannot perform data cleaning.")

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
        if args.drop_columns == ["[]"]:
            args.drop_columns = []

        # Print debugging information after processing
        print("Value of args.drop_columns after processing:", args.drop_columns)

        # Clean data
        df_cleaned = clean_data(df, args.drop_columns)

        # Log cleaned data as output artifact
        output_artifact = wandb.Artifact(
            args.output_artifact_name,
            type=args.output_artifact_type,
            description=args.output_artifact_description,
        )
        with output_artifact.new_file("cleaned_data.csv", mode="w") as f:
            df_cleaned.to_csv(f, index=False)

        wandb.log_artifact(output_artifact)
        logger.info("Data cleaned and artifact logged to Weights & Biases")

    finally:
        # Cleanup: Delete the local file downloaded from wandb
        os.remove(artifact_path)
        # Additionally, if you want to delete the entire directory:
        dir_to_remove = Path(artifact_path).parents[1]
        if (
            dir_to_remove.exists()
        ):  # Ensure the directory exists before trying to delete it
            shutil.rmtree(dir_to_remove)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean data and log to wandb")
    parser.add_argument(
        "--input_artifact", type=str, required=True, help="Name for the input artifact"
    )
    parser.add_argument(
        "--drop_columns", type=str, nargs="*", help="List of column names to drop"
    )
    parser.add_argument(
        "--output_artifact_name",
        type=str,
        required=True,
        help="Name for the output artifact",
    )
    parser.add_argument(
        "--output_artifact_type",
        type=str,
        required=True,
        help="Type of the output artifact",
    )
    parser.add_argument(
        "--output_artifact_description",
        type=str,
        help="Description for the output artifact",
    )

    args = parser.parse_args()
    main(args)
