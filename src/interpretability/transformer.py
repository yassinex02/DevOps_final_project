import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FactorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.apply(lambda col: pd.factorize(col)[0])
        return X_transformed