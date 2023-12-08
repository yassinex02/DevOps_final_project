"""
This script splits the provided dataframe into training and test sets and logs them back to wandb.
"""

import argparse
import logging
import pandas as pd
from typing import Tuple
import wandb
import tempfile
from sklearn.model_selection import train_test_split


# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame]:
    
    """
    Split the data into training and test sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the data to use as the test set.
        random_state (int): The random state to use for reproducibility.
    
    Returns: 
        Tuple[pd.DataFrame]: The training and test sets.
    """

    try:
        logger.info("Splitting data.")
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Data splitting failed: %s", e)
        raise


def main(args):
    """
    Main function to execute the data splitting process and log the results as artifacts in wandb.

    Args:
        args (argparse.Namespace): Command line arguments.
    """

    try:
        # Initialize the wandb run
        wandb.init(job_type="train_val_test_split")

        # Fetching and logging the input artifact
        logger.info(f"Fetching artifact {args.input}")
        artifact_local_path = wandb.use_artifact(args.input).file()
        df = pd.read_csv(artifact_local_path)

        # Splitting the data
        X_train, X_test, y_train, y_test = split_data(
            df, args.target_column, args.test_size, args.random_state
        )
        
        # Logging the split datasets as artifacts
        datasets = {
            "X_train": X_train, "X_test": X_test, 
            "y_train": y_train, "y_test": y_test
        }
        for dataset_name, dataset in datasets.items():
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                dataset.to_csv(tmp.name, index=False)
                temp_file_path = tmp.name

                # Create and log the artifact
                artifact = wandb.Artifact(
                    name=f"{dataset_name}_data",
                    type="dataset",
                    description=f"{dataset_name} split of the dataset"
                )
                artifact.add_file(temp_file_path, name=f"{dataset_name}.csv")
                logger.info(f"Logging {dataset_name} dataset as a W&B artifact.")
                wandb.log_artifact(artifact)
                artifact.wait()

        logger.info("Data splitting and logging process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training and testing sets and log to W&B"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input artifact name to split"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column in the dataset"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        required=True,
        help="Size of the test split as a fraction of the dataset"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for random number generator"
    )

    args = parser.parse_args()
    main(args)


         
     
      

