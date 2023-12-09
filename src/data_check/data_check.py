""" This module contains functions for checking the data with series of tests."""

import pandas as pd
import numpy as np
import logging
import scipy.stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Non-Deterministic Test
def test_kolmogorov_smirnov(ref_data, data, ks_alpha, numerical_columns):
    """Test that the distribution of the numerical columns in the data is the same as in the reference data."""

    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:
        ts, p_value = scipy.stats.ks_2samp(
            ref_data[col],
            data[col],
            alternative='two-sided'
        )
        assert p_value > alpha_prime, f"Column {col} failed the KS test with p-value {p_value}"


# Deterministic Test
def test_column_presence_and_type(data, required_columns):
    """Test that the data has the required columns and that they are of the required type."""

    logger.info("Testing that the data has the required columns and that they are of the required type.")

    # Function to map string to pandas data type check function
    def map_type_check_function(type_str):
        if type_str == "int64":
            return pd.api.types.is_int64_dtype
        elif type_str == "float64":
            return pd.api.types.is_float64_dtype
        elif type_str == "object":
            return pd.api.types.is_object_dtype
        else:
            raise ValueError(f"Unsupported data type: {type_str}")

    # Convert the required_columns with string data types to pandas data type check functions
    converted_required_columns = {col: map_type_check_function(dtype_str) for col, dtype_str in required_columns.items()}

    # Check column presence
    assert set(data.columns.values).issuperset(set(converted_required_columns.keys())), "Some required columns are missing."

    # Check column data types
    for col_name, type_check_function in converted_required_columns.items():
        assert type_check_function(data[col_name]), f"Column {col_name} is not of type {required_columns[col_name]}"

# Deterministic Test
def test_class_names(data, known_classes):
    """Test that the data has the required classes."""

    logger.info("Testing that the data has the required classes.")
   
    assert data["default"].isin(known_classes).all()

def test_missing_values(data):
    """Test that the data has no missing values."""
    logger.info("Testing for missing values...")
    assert not data.isnull().values.any(), "There are missing values in the data."

# Deterministic Test
def test_column_ranges(data, ranges):
    """Test that the data has the required ranges."""

    logger.info("Testing that the data has the required ranges.")
    
    # Convert ranges from list format to tuple format
    converted_ranges = {col: tuple(range_vals) for col, range_vals in ranges.items()}

    for col_name, (minimum, maximum) in converted_ranges.items():
        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )
       

