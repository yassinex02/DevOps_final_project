""" 
Test data loading module.
"""
import os
import pandas as pd
from src.load_data.data_load import load_data


def test_load_excel_file():
    """Test loading data from excel file."""

    file_path = "tests/data/test_data.xlsx"
    df = load_data(file_path, sheet_name="Sheet1")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Name", "Age", "Gender"]


def test_load_csv_file():
    """Test loading data from csv file."""

    file_path = "tests/data/test_data.csv"
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Name", "Age", "Gender"]


def test_load_json_file():
    """Test loading data from json file."""

    file_path = "tests/data/test_data.json"
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Name", "Age", "Gender"]


def test_file_not_found():
    """Test loading data from non-existent file."""

    file_path = "tests/data/non_existent_file.xlsx"
    try:
        df = load_data(file_path)
    except FileNotFoundError as e:
        assert str(e) == f"File not found: {file_path}"


def test_unsupported_file_format():
    """Test loading data from unsupported file format."""

    file_path = "tests/data/test_data.txt"
    try:
        df = load_data(file_path)
    except ValueError as e:
        assert str(e) == "Unsupported file format: .txt"
