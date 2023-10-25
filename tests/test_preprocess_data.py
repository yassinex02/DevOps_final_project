import pandas as pd
import numpy as np
from src.preprocessing.preprocess_data import preprocess_data

def test_preprocess_data():
    # Create a sample dataframe
    df = pd.DataFrame({
        'ed': ['HS', 'College', 'Graduate'],
        'default': [0, 1, np.nan],
        'balance': [1000, 2000, 3000]
    })

    # Preprocess the dataframe
    df = preprocess_data(df)

    # Check that the 'ed' and 'default' columns are of type 'category'
    assert df['ed'].dtype.name == 'category'
    assert df['default'].dtype.name == 'category'

    # Check that the 'balance' column is of type 'int64'
    assert df['balance'].dtype.name == 'int64'

    # Check that the missing value in the 'default' column has been handled
    assert df['default'].isna().sum() == 0