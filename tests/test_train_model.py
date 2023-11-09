from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.train_model import (
    evaluate_model,
    perform_baselining,
    perform_model_building,
    plot_roc_curve,
    random_undersampling,
    save_model,
    split_data,
    train_logistic_regression,
)


# Fixture for creating a dummy dataset
@pytest.fixture
def dummy_data():
    data = {"feature1": [0, 1, 0, 1], "feature2": [1, 0, 1, 0], "target": [0, 1, 0, 1]}
    return pd.DataFrame(data)


# Test for split_data function


def test_split_data(dummy_data):
    X_train, X_test, y_train, y_test = split_data(dummy_data, "target", 0.5, 1)
    assert len(X_train) == len(y_train) == 2
    assert len(X_test) == len(y_test) == 2


# Test for random_undersampling function


def test_random_undersampling(dummy_data):
    X = dummy_data.drop(["target"], axis=1)
    y = dummy_data["target"]
    X_resampled, y_resampled = random_undersampling(X, y, 1)
    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.sum() == len(y_resampled) / 2  # Check for balanced classes


# Test for perform_baselining function


def test_perform_baselining(dummy_data):
    _, X_test, _, y_test = split_data(dummy_data, "target", 0.5, 1)
    baseline_df = perform_baselining(X_test, y_test)
    assert "Majority Class" in baseline_df.columns
    assert "Stratified Random" in baseline_df.columns


# Test for train_logistic_regression function


def test_train_logistic_regression(dummy_data):
    X_train, _, y_train, _ = split_data(dummy_data, "target", 0.5, 1)
    model = train_logistic_regression(X_train, y_train, 1)
    assert isinstance(model, LogisticRegression)


# Test for evaluate_model function


def test_evaluate_model(dummy_data):
    X_train, X_test, y_train, y_test = split_data(dummy_data, "target", 0.5, 1)
    model = train_logistic_regression(X_train, y_train, 1)
    evaluation_df = evaluate_model(model, X_test, y_test)
    assert "Accuracy" in evaluation_df.columns
    assert "Precision" in evaluation_df.columns
    assert "Recall" in evaluation_df.columns
    assert "ROC AUC" in evaluation_df.columns


# Test for plot_roc_curve function with mock to avoid plotting


@patch("matplotlib.pyplot.savefig")
def test_plot_roc_curve(mock_savefig, dummy_data):
    X_train, X_test, y_train, y_test = split_data(dummy_data, "target", 0.5, 1)
    model = train_logistic_regression(X_train, y_train, 1)
    plot_roc_curve(model, X_test, y_test, "reports/")
    mock_savefig.assert_called_once()


# Test for save_model function with mock to avoid file writing


@patch("joblib.dump")
def test_save_model(mock_dump, dummy_data):
    model = LogisticRegression()
    save_model(model, "dummy_path.joblib")
    mock_dump.assert_called_once_with(model, "dummy_path.joblib")


# Test for perform_model_building function with mock to avoid file operations


@patch("src.models.train_model.save_model")
@patch("src.models.train_model.plot_roc_curve")
@patch(
    "src.models.train_model.evaluate_model",
    return_value=pd.DataFrame({"Accuracy": [0.5]}),
)
@patch(
    "src.models.train_model.train_logistic_regression",
    return_value=LogisticRegression(),
)
@patch(
    "src.models.train_model.perform_baselining",
    return_value=pd.DataFrame({"Majority Class": [0.5]}),
)
@patch("src.models.train_model.random_undersampling")
@patch("src.models.train_model.split_data")
def test_perform_model_building(
    mock_split_data,
    mock_random_undersampling,
    mock_perform_baselining,
    mock_train_logistic_regression,
    mock_evaluate_model,
    mock_plot_roc_curve,
    mock_save_model,
    dummy_data,
):
    # Mocking the split data function to return the dummy data directly
    mock_split_data.return_value = (
        dummy_data.drop("target", axis=1),
        dummy_data.drop("target", axis=1),
        dummy_data["target"],
        dummy_data["target"],
    )
    mock_random_undersampling.return_value = (
        pd.DataFrame({"feature1": [0, 1], "feature2": [1, 0]}),
        pd.Series([0, 1]),
    )
    # Call the perform_model_building function with the dummy data
    perform_model_building(dummy_data, "target", 0.5, 1)

    # Check if all mocked functions were called
    mock_split_data.assert_called_once()
    mock_random_undersampling.assert_called_once()
    mock_perform_baselining.assert_called_once()
    mock_train_logistic_regression.assert_called_once()
    mock_evaluate_model.assert_called_once()
    mock_plot_roc_curve.assert_called_once()
    mock_save_model.assert_called_once()
