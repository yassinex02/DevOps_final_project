# -*- coding: utf-8 -*-
#%%
import yaml
import logging
import os
import pandas as pd
from preprocess_data import preprocess_data
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


project_root = Path(__file__).resolve().parents[2]
#%%Get the root directory (parent of the current script's directory in this case)

# Construct the path to config.yaml
config_path = project_root / "config.yaml"

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

#%%
def main(config):
    """Runs data processing scripts to turn raw data from (../raw) into
       cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('preprocessing data')

    # Get absolute paths
    raw_filepath = project_root / config['data']['raw_filepath']
    processed_filepath = project_root / config['data']['processed_filepath']

    # Load the data
    df = pd.read_csv(raw_filepath)

    # Preprocess the data
    df = preprocess_data(df)

    # Save the data
    df.to_csv(processed_filepath, index=False)

    logger.info(f"Preprocessed data saved to {processed_filepath}")



if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(config)
    


# %%
