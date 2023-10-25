import pandas as pd
import logging
import os
from pathlib import Path
import yaml

# Initialize logging


# current_script_directory = Path(__file__).parent

# # Get the root directory (parent of the current script's directory in this case)
# root_directory = current_script_directory.parent

# # Construct the path to config.yaml
# config_path = root_directory / "config.yaml"

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)


logging.basicConfig(level=logging.getLevelName(config['logging']['level']),
                    format=config['logging']['format'])

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
            df = pd.read_json(file_path, orient='split')
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logging.info(f"Successfully loaded data from {file_path}")
        return df
    
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise


