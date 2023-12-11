import pandas as pd
import pytest
from src.preprocessing.preprocess_data import clean_data

# Your test functions will be here


def test_clean_data_empty_dataframe():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame())


def test_clean_data_drop_columns():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    columns_to_drop = ['B']

    # Call clean_data and assert the dropped column is not in the DataFrame
    cleaned_df = clean_data(df, drop_columns=columns_to_drop)
    assert 'B' not in cleaned_df.columns
