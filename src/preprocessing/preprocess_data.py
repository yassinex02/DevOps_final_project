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

# Ensure that only the columns above are present in the dataframe

def check_columns():
    """Check if the columns in the dataframe are the expected columns."""
    try:
        logging.info("Checking columns.")
        expected_columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']
        if set(expected_columns) == set(df.columns):
            logging.info("Columns are correct.")
        else:
            logging.error("Columns are not correct.")
            raise
    except Exception as e:
        logging.error(f"An error occurred while checking the columns: {e}")
        raise






