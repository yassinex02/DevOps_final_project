import argparse
import logging
import os
import pandas as pd
import wandb

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load data from an Excel, CSV, or JSON file into a DataFrame.

    Parameters:
        file_path (str): The path to the data file.
        sheet_name (str, optional): The name of the sheet to read if the file is an Excel file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)

    # Load the data based on file extension
    try:
        if file_extension == '.xlsx':
            logging.info(f"Loading Excel file from {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension == '.csv':
            logging.info(f"Loading CSV file from {file_path}")
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            logging.info(f"Loading JSON file from {file_path}")
            df = pd.read_json(file_path)
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            raise ValueError(f"Unsupported file format: {file_extension}")

        logging.info(f"Successfully loaded data from {file_path}")
        return df

    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise

def main(args):
    """
    Load data from a file and log it to Weights & Biases.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    wandb.init(job_type="data_loader")

    df = load_data(args.file_path, args.sheet_name)

    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    with artifact.new_file("raw_data.csv", mode="w") as f:
        df.to_csv(f, index=False)

    wandb.log_artifact(artifact)
    logger.info("Data loaded and artifact logged to Weights & Biases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data and log to wandb")
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the data file"
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        help="Sheet name if Excel file"
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        required=True,
        help="Name for the W&B artifact"
    )
    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the artifact"
    )
    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact"
    )

    args = parser.parse_args()
    main(args)

