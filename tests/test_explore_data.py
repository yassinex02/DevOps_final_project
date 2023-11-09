from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from pandas_profiling import ProfileReport
import src.exploration.explore_data as ed


# Create a fixture to mock the config and EXPECTED_COLUMNS
@pytest.fixture
def mock_config():
    mock_data = {
        'logging': {
            'level': 'info',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'expected_columns': ['id', 'name', 'age'],
        'profile_report': {
            'title': 'Test Profiling Report'
        }
    }
    with patch('src.exploration.explore_data.read_yaml_config', return_value=mock_data):
        ed.config = ed.read_yaml_config("config.yaml")
        ed.EXPECTED_COLUMNS = ed.config.get('expected_columns', [])
        yield

def test_generate_profiling_report(mock_config):
    # Create a dummy DataFrame
    data = {'id': [1, 2], 'name': ['Alice', 'Bob'], 'age': [30, 40]}
    df = pd.DataFrame(data)
    
    # Patch the ProfileReport to not actually generate a report during tests
    with patch.object(ProfileReport, 'to_file', return_value=None) as mock_method:
        # Test with correct DataFrame and columns
        ed.generate_profiling_report(df, 'dummy_output.html')
        assert mock_method.called

    # Test with empty DataFrame
    with pytest.raises(ValueError):
        ed.generate_profiling_report(pd.DataFrame(), 'dummy_output.html')

    # Test with missing expected columns
    missing_column_data = {'id': [1, 2], 'name': ['Alice', 'Bob']}  # age is missing
    df_missing_columns = pd.DataFrame(missing_column_data)
    with pytest.raises(ValueError):
        ed.generate_profiling_report(df_missing_columns, 'dummy_output.html')
