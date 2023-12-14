"""
This module contains the unit tests for the split_train_test.py script.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_split.split_train_test import main, split_data


@pytest.fixture
def sample_dataframe():
    """
    Fixture that returns a sample dataframe for testing.
    """
    return pd.DataFrame(
        {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "target": [0, 1, 0, 1]}
    )


def test_split_data_success(sample_dataframe):
    """
    Test case to check the success scenario of the split_data function.
    """
    test_size = 0.25
    target_column = "target"
    random_state = 42
    X_train, X_test, y_train, y_test = split_data(
        sample_dataframe, target_column, test_size, random_state
    )

    # Assertions to check the split
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
    assert target_column not in X_train.columns
    assert target_column not in X_test.columns


def test_split_data_invalid_test_size(sample_dataframe):
    """
    Test case to check the scenario when an
    invalid test_size is provided to the split_data function.
    """
    test_size = -1  # Invalid test_size
    target_column = "target"
    random_state = 42
    with pytest.raises(ValueError):
        split_data(sample_dataframe, target_column, test_size, random_state)


def test_split_data_invalid_target_column(sample_dataframe):
    """
    Test case to check the scenario when an invalid target_column 
    is provided to the split_data function.
    """
    test_size = 0.25
    target_column = "invalid_column"  # Invalid target_column
    random_state = 42
    with pytest.raises(KeyError):
        split_data(sample_dataframe, target_column, test_size, random_state)


@patch("src.data_split.split_train_test.wandb")
def test_main_process(mock_wandb, sample_dataframe):
    """
    Test case to check the main process of the script.
    """
    args = MagicMock(
        input="input_artifact.csv",
        target_column="target",
        test_size=0.25,
        random_state=42,
    )

    # Mocking wandb.init(), wandb.use_artifact(), and pd.read_csv() to use the sample dataframe
    mock_wandb.init.return_value = None
    mock_wandb.use_artifact.return_value = MagicMock(
        file=MagicMock(return_value="input_artifact.csv")
    )

    with patch("pandas.read_csv", return_value=sample_dataframe), patch(
        "tempfile.NamedTemporaryFile"
    ) as mock_temp_file, patch("pandas.DataFrame.to_csv"):
        mock_temp_file.return_value.__enter__.return_value.name = "temp_file.csv"
        main(args)

    # Verify wandb and other interactions
    mock_wandb.init.assert_called_once_with(job_type="train_val_test_split")
    mock_wandb.use_artifact.assert_called_once_with("input_artifact.csv")
    mock_wandb.log_artifact.assert_called()
