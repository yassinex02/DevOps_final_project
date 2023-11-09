import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.interpretability.model_interpretability import (
    extract_feature_importance, run_lime_analysis, run_shap_analysis)


def test_run_shap_analysis(mocker):
    # Mock data
    mock_model = LogisticRegression()
    mock_X_test = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    instance_index = 1
    
    # Mock SHAP and Matplotlib functions
    mocker.patch('shap.initjs')
    mocker.patch('shap.Explainer')
    mocker.patch('shap.summary_plot')
    mocker.patch('shap.force_plot')
    mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('matplotlib.pyplot.close')
    
    # Run the function
    run_shap_analysis(mock_model, mock_X_test, instance_index)

def test_run_lime_analysis(mocker):
    # Mock data
    mock_model = LogisticRegression()
    mock_X_test = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    mock_y_test = pd.Series([0, 1, 0])
    instance_index = 1

    # Mock LIME functions
    mocker.patch('lime.lime_tabular.LimeTabularExplainer')
    mocker.patch('lime.explanation.Explanation.save_to_file')

    # Run the function
    run_lime_analysis(mock_model, mock_X_test, mock_y_test, instance_index)

def test_extract_feature_importance(mocker):
    # Mock data
    mock_model = LogisticRegression()
    mock_model.coef_ = np.array([[0.1, 0.2]])
    mock_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})

    # Mock pandas DataFrame.to_csv
    mocker.patch('pandas.DataFrame.to_csv')

    # Run the function
    extract_feature_importance(mock_model, mock_X)