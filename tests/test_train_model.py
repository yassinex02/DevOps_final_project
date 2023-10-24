import pandas as pd
import numpy as np
import pytest
from src.models import train_model as model_building

# Sample DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'default': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Sample parameters
test_size = 0.2
random_state = 42

def test_split_data():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1
model_building.
def test_train_logistic_regression():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    model = model_building.train_logistic_regression(X_train, y_train, random_state)
    assert model is not None

def test_train_random_forest():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    model = model_building.train_random_forest(X_train, y_train, random_state)
    assert model is not None

def test_train_stochastic_gradient_descent():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    model = model_building.train_stochastic_gradient_descent(X_train, y_train, random_state)
    assert model is not None

def test_train_support_vector_machine():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    model = model_building.train_support_vector_machine(X_train, y_train, random_state)
    assert model is not None

def test_evaluate_model():
    X_train, X_test, y_train, y_test = model_building.split_data(df, test_size, random_state)
    model = model_building.train_logistic_regression(X_train, y_train, random_state)
    metrics = model_building.evaluate_model(model, X_test, y_test)
    assert 'Accuracy' in metrics.columns
    assert 'Precision' in metrics.columns
    assert 'Recall' in metrics.columns
    assert 'Specificity' in metrics.columns
    assert 'F1 Score' in metrics.columns
    assert 'PPV' in metrics.columns
    assert 'NPV' in metrics.columns
    assert 'ROC AUC' in metrics.columns
    assert 'Confusion Matrix' in metrics.columns

def test_perform_model_building():
    metrics = model_building.perform_model_building(df, "logistic_regression", "default", test_size, random_state)
    assert 'Accuracy' in metrics.columns
    assert 'Precision' in metrics.columns
    assert 'Recall' in metrics.columns
    assert 'Specificity' in metrics.columns
    assert 'F1 Score' in metrics.columns
    assert 'PPV' in metrics.columns
    assert 'NPV' in metrics.columns
    assert 'ROC AUC' in metrics.columns
    assert 'Confusion Matrix' in metrics.columns

