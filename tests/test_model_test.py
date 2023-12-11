import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.model_tests.model_testing import evaluate_model


def test_evaluate_model():
    # Sample data
    y_test = pd.Series([0, 1, 0, 1])
    y_pred = pd.DataFrame([0, 1, 0, 1])
    y_prob = pd.DataFrame([0.25, 0.75, 0.25, 0.75])

    # Call the evaluate_model function
    result = evaluate_model(y_test, y_pred, y_prob)

    # Assertions to check if the metrics are calculated correctly
    assert result['Accuracy'][0] == 1.0
    assert result['Precision'][0] == 1.0
    assert result['Recall'][0] == 1.0
    assert result['F1 Score'][0] == 1.0


