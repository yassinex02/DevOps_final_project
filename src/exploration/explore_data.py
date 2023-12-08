import argparse
import logging
from typing import Optional
import pandas as pd
from ydata_profiling import ProfileReport
import wandb


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)



def generate_profiling_report(df: pd.DataFrame,report_title: str ,output_file: str, expected_columns: list, columns: Optional[list] = None):
    """
    Generate a profiling report from a DataFrame using ydata-profiling's ProfileReport.
    ... [rest of the docstring]
    """
    logging.info("Generating profiling report...")

    # Check if expected columns exist in the DataFrame
    missing_expected_columns = [col for col in expected_columns if col not in df.columns]
    if missing_expected_columns:
        logging.error(f"Expected columns not found in the DataFrame: {missing_expected_columns}")
        raise ValueError(f"Expected columns not found in the DataFrame: {missing_expected_columns}")

    # Select specified columns if provided
    selected_df = df[columns] if columns else df

    # Generate profiling report
    try:
        profile = ProfileReport(selected_df, title=report_title)
        profile.to_file(output_file)
        logging.info(f"Profiling report generated and saved to {output_file}.")
    except Exception as e:
        logging.error(f"An error occurred while generating the profiling report: {e}")
        raise


def main(args):

    wandb.init(job_type="data_exploration")

    artifact = wandb.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Load data
    try:
        df = pd.read_csv(artifact_path)

        # Convert the string '[]' to an actual empty list
        if args.columns == ['[]']:
            args.columns = []
        
        # Split comma-separated strings into lists
        args.expected_columns = args.expected_columns[0].split(' ')
        args.columns = args.columns[0].split(' ')

        # Generate profiling report
        generate_profiling_report(df, args.report_title, args.output_file, args.expected_columns, args.columns)

        # Log the profiling report as an artifact
        output_artifact = wandb.Artifact(
            name=args.output_artifact_name,
            type=args.output_artifact_type,
            description=args.output_artifact_description,
        )
        output_artifact.add_file(args.output_file)
        wandb.log_artifact(output_artifact)

        logger.info("Profiling report generated and logged to Weights & Biases.")
    except Exception as e:
        logger.error(f"An error occurred in main function: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and log a data profiling report to Weights & Biases.")
    parser.add_argument("--input_artifact", type=str, required=True, help="Weights & Biases artifact path for the input data.")
    parser.add_argument("--report_title", type=str, required=True, help="Title of the profiling report.")
    parser.add_argument("--output_file", type=str, required=True, help="File path for the output profiling report.")
    parser.add_argument("--expected_columns", type=str, nargs='*', required=True, help="List of expected columns in the DataFrame.")
    parser.add_argument("--columns", type=str, nargs='*', help="List of columns to include in the report, if not all.")
    parser.add_argument("--output_artifact_name", type=str, required=True, help="Name for the output artifact in Weights & Biases.")
    parser.add_argument("--output_artifact_type", type=str, required=True, help="Type for the output artifact in Weights & Biases.")
    parser.add_argument("--output_artifact_description", type=str, help="Description for the output artifact in Weights & Biases.")

    args = parser.parse_args()
    main(args)
            
            


