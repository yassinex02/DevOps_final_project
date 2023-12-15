"""
This module contains the unit tests for the preprocess_data module.
"""
import pandas as pd
import pytest
from src.preprocessing.preprocess_data import clean_data



def test_clean_data_empty_dataframe():
    """
    Test case for clean_data function with an empty DataFrame.
    It should raise a ValueError.
    """
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame())


def test_clean_data_drop_columns():
    """
    Test case for clean_data function with dropping columns.
    It should drop the specified columns from the DataFrame.
    """
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
