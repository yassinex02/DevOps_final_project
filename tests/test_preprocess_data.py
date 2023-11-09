import pandas as pd
import pytest
from src.preprocessing.preprocess_data import clean_data, factorize_column, standardize_columns

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

def test_factorize_column():
    # Create a sample DataFrame
    df = pd.DataFrame({'Category': ['apple', 'orange', 'apple', 'banana']})
    column_to_factorize = 'Category'
    
    # Call factorize_column and assert that the column has been factorized
    factorized_df = factorize_column(df, column_to_factorize)
    assert factorized_df['Category'].nunique() == 3
    assert all(isinstance(x, int) for x in factorized_df['Category'])

def test_standardize_columns():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'Value1': [1, 2, 3],
        'Value2': [4, 5, 6]
    })
    columns_to_standardize = ['Value1', 'Value2']
    
    # Call standardize_columns and assert that the columns have been standardized
    standardized_df = standardize_columns(df, columns_to_standardize)
    assert standardized_df['Value1'].mean() == pytest.approx(0)
    assert standardized_df['Value2'].mean() == pytest.approx(0)
    assert standardized_df['Value1'].std(ddof=0) == pytest.approx(1)
    assert standardized_df['Value2'].std(ddof=0) == pytest.approx(1)
