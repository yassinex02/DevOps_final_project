import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.exploration.explore_data import generate_profiling_report

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })

@pytest.fixture
def profile_report_mock():
    profile_mock = MagicMock()
    profile_mock.to_file = MagicMock()
    return profile_mock

def test_generate_profiling_report_success(sample_dataframe, profile_report_mock):
    # Test for successful report generation
    with patch('src.exploration.explore_data.ProfileReport', return_value=profile_report_mock):
        generate_profiling_report(sample_dataframe, "Test Report", "test_report.html", ['col1', 'col2'])
        assert profile_report_mock.to_file.called

def test_generate_profiling_report_with_selected_columns(sample_dataframe, profile_report_mock):
    # Test for report generation with selected columns
    with patch('src.exploration.explore_data.ProfileReport', return_value=profile_report_mock):
        generate_profiling_report(sample_dataframe, "Test Report", "test_report.html", ['col1', 'col2'], ['col1'])
        assert profile_report_mock.to_file.called

def test_generate_profiling_report_missing_columns(sample_dataframe):
    # Test for missing columns in DataFrame
    with pytest.raises(ValueError) as excinfo:
        generate_profiling_report(sample_dataframe, "Test Report", "test_report.html", ['col1', 'col3'])
    assert "Expected columns not found in the DataFrame" in str(excinfo.value)

def test_generate_profiling_report_exception(sample_dataframe):
    # Test to simulate exception during report generation
    with patch('src.exploration.explore_data.ProfileReport', side_effect=Exception("Test Error")):
        with pytest.raises(Exception) as excinfo:
            generate_profiling_report(sample_dataframe, "Test Report", "test_report.html", ['col1', 'col2'])
        assert "Test Error" in str(excinfo.value)


