import pandas as pd
import logging

# initialize logging
logging.basicConfig(level=logging.INFO)


def preprocess_data(df):
    """Preprocess the dataframe: convert columns to appropriate types, handle missing values etc."""
    logging.info("Preprocessing the data.")

    try:
        for col in ['ed', 'default']:
            df[col] = df[col].astype('category')
        return df
    except Exception as e:
        logging.error(f"An error occurred while preprocessing the data: {e}")
        raise
