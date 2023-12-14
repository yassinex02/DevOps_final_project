"""
This module contains the FactorizeTransformer class.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FactorizeTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies factorization to the input data.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        X (pandas.DataFrame): The input data to fit the transformer on.
        y (optional): The target variable (ignored).

        Returns:
        self: The fitted transformer object.
        """
        return self

    def transform(self, x):
        """
        Transform the input data.

        Parameters:
        X (pandas.DataFrame): The input data to transform.

        Returns:
        X_transformed (pandas.DataFrame): The transformed data.
        """

        x_transformed = x.apply(lambda col: pd.factorize(col)[0])

        return x_transformed
