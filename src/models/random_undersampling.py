import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple
from imblearn.under_sampling import RandomUnderSampler



def random_undersampling(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Tuple[pd.DataFrame]:
    """Split data into training and testing sets."""
    try:
        logging.info("Random undersampling.")
        rus = RandomUnderSampler(random_state=random_state)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        return X_train_rus, y_train_rus
    except Exception as e:
        logging.error(f"An error occurred while random undersampling the data: {e}")
        raise

