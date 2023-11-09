
from typing import Optional
import pandas as pd
from ydata_profiling import ProfileReport
import logging
import yaml
import argparse

# Function to read YAML configuration


def read_yaml_config(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to read the YAML configuration file: {e}")


# Read the YAML configuration file
config_file_path = "config.yaml"
config = read_yaml_config(config_file_path)

# Initialize logging
logging.basicConfig(
    format=config['logging']['format'], level=config['logging']['level'].upper())

EXPECTED_COLUMNS = config.get('expected_columns', [])


def generate_profiling_report(df: pd.DataFrame, output_file: str, columns: Optional[list] = None):
    """
    Generate a profiling report from a DataFrame using ydata-profiling's ProfileReport.

    Parameters:
        df (pd.DataFrame): The DataFrame to profile.
        output_file (str): The path to the file where the report will be saved.
        columns (list, optional): List of column names to include in the profiling report. If None, all columns are included.

    Returns:
        None

    Raises:
        ValueError: If the DataFrame is empty or if specified columns are not present.
    """
    # Get title from config
    report_title = config.get('profile_report', {}).get(
        'title', 'Profiling Report')
    logging.info("Generating profiling report...")

    # Validate input DataFrame
    if df.empty:
        logging.error(
            "Input DataFrame is empty. Cannot generate profiling report.")
        raise ValueError(
            "Input DataFrame is empty. Cannot generate profiling report.")

    # Check if expected columns exist in the DataFrame
    missing_expected_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_expected_columns:
        logging.error(
            f"Expected columns not found in the DataFrame: {missing_expected_columns}")
        raise ValueError(
            f"Expected columns not found in the DataFrame: {missing_expected_columns}")

    # Select specified columns if provided
    selected_df = df[columns] if columns else df

    # Generate profiling report
    try:
        profile = ProfileReport(selected_df, title=report_title)
        profile.to_file(output_file)
        logging.info(f"Profiling report generated and saved to {output_file}.")
    except Exception as e:
        logging.error(
            f"An error occurred while generating the profiling report: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate a profiling report from a DataFrame.')
    parser.add_argument('input_filepath', type=str,
                        help='Path to the input CSV file')
    parser.add_argument('output_filepath', type=str,
                        help='Path to save the profiling report')
    parser.add_argument('--columns', type=str, default=None,
                        help='List of column names to include in the profiling report. If not provided, all columns are included.')

    args = parser.parse_args()

    # Read the DataFrame from the input file or any other source
    df = pd.read_csv(args.input_filepath)

    # Split the columns string into a list
    columns_list = args.columns.split(',') if args.columns else None

    # Call the generate_profiling_report function with the DataFrame and other arguments
    generate_profiling_report(df, args.output_filepath, columns_list)


if __name__ == "__main__":
    main()
