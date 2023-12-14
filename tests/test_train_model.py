"""
This module contains unit tests for the train_model module.
"""
import json
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from src.models.train_model import (
    evaluate_model,
    get_preprocessing_pipeline,
    main,
    random_undersampling,
    train_logistic_regression,
)


def test_get_preprocessing_pipeline():
    """
    Test for the get_preprocessing_pipeline function.
    """
    numerical_cols = ["num1", "num2"]
    factorize_cols = ["fact1", "fact2"]
    pipeline = get_preprocessing_pipeline(numerical_cols, factorize_cols)

    # Get names of transformers
    transformer_names = [name for name, _, _ in pipeline.transformers]

    assert "num" in transformer_names
    assert "fact" in transformer_names


def test_random_undersampling():
    """
    Test for the random_undersampling function.
    """
    X_train = pd.DataFrame({"feature": [1, 2, 1, 2]})
    y_train = pd.Series([0, 1, 0, 1])
    random_state = 42

    X_train_rus, y_train_rus = random_undersampling(
        X_train, y_train, random_state)

    assert len(X_train_rus) == len(y_train_rus)


def test_train_logistic_regression():
    """
    Test for the train_logistic_regression function.
    """
    X, y = make_classification(n_samples=100, n_features=4)
    hyperparameters = {"max_iter": 100}

    model = train_logistic_regression(X, y, hyperparameters)

    assert model.classes_ is not None


def test_evaluate_model():
    """
    Test for the evaluate_model function.
    """
    y_test = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1])

    results = evaluate_model(y_test, y_pred)

    assert results["Accuracy"].iloc[0] == accuracy_score(y_test, y_pred)


# Sample arguments for testing
sample_args = {
    "X_train_artifact": "path/to/X_train.csv",
    "y_train_artifact": "path/to/y_train.csv",
    "val_size": 0.2,
    "numerical_cols": "col1 col2",  # make sure these columns exist in mock_X_train
    "factorize_cols": "col3",  # add this column to mock_X_train
    "hyperparameters": "path/to/hyperparameters.json",
    "model_artifact": "model_artifact_name",
    "random_seed": 42,
}

# Sample data for the hyperparameters file
sample_hyperparameters = {"max_iter": 100, "solver": "liblinear"}

# Convert 'col3' to numerical categories for the test
mock_X_train = pd.DataFrame(
    {
        "col1": [1, 2, 3, 4, 5, 6],
        "col2": [5, 6, 7, 8, 9, 10],
        "col3": pd.Categorical(
            ["a", "b", "c", "a", "b", "c"]
        ).codes,  # Convert to numerical categories
    }
)
mock_y_train = pd.Series([0, 1, 0, 1, 0, 1])


@patch("src.models.train_model.argparse.ArgumentParser.parse_args")
@patch("src.models.train_model.pd.read_csv", side_effect=[mock_X_train, mock_y_train])
@patch("src.models.train_model.json.load", return_value=sample_hyperparameters)
@patch("src.models.train_model.wandb")
@patch("src.models.train_model.mlflow")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(sample_hyperparameters),
)
def test_main(
    mock_file_open,
    mock_mlflow,
    mock_wandb,
    mock_json_load,
    mock_read_csv,
    mock_parse_args,
):
    """
    Test for the main function.
    """
    # Mock the command line arguments
    mock_args = MagicMock()
    mock_args.X_train_artifact = sample_args["X_train_artifact"]
    mock_args.y_train_artifact = sample_args["y_train_artifact"]
    mock_args.val_size = sample_args["val_size"]
    mock_args.numerical_cols = sample_args["numerical_cols"].split()
    mock_args.factorize_cols = sample_args["factorize_cols"].split()
    mock_args.hyperparameters = sample_args["hyperparameters"]
    mock_args.model_artifact = sample_args["model_artifact"]
    mock_args.random_seed = sample_args["random_seed"]
    mock_parse_args.return_value = mock_args

    # Call the main function
    main(mock_args)

    # Assertions to ensure the main function is calling the expected methods
    mock_read_csv.assert_called()
    mock_json_load.assert_called()
    mock_wandb.init.assert_called()
    mock_mlflow.sklearn.save_model.assert_called()
    mock_file_open.assert_called_with(sample_args["hyperparameters"])
