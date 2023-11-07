import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(format=config['logging']['format'], level=config['logging']['level'].upper())

def split_data(df:pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame]:
    """Split data into training and testing sets."""
    try:
        logging.info("Splitting the data into training and test sets.")
        X = df.drop(['default'], axis=1)
        y = df['default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"An error occurred while splitting the data: {e}")
        raise
    # X = df[df.columns.drop('default')].values
    # y = df["default"].values
    # return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def random_undersampling(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Tuple[pd.DataFrame]:
    """Split data into training and testing sets."""
    try:
        logging.info("Random undersampling.")
        rus = RandomUnderSampler(random_state=random_state)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        return X_train_rus, y_train_rus
    except Exception as e:
        logging.error(f"An error occurred while random undersampling the data: {e}")
        raise

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, random_state: int, class_weight=None) -> LogisticRegression:
    """Train a logistic regression model."""
    try:
        logging.info("Training a logistic regression model.")
        lr_model = LogisticRegression(random_state=random_state, class_weight=class_weight)
        lr_model.fit(X_train, y_train)
        return lr_model
    except Exception as e:
        logging.error(f"An error occurred while training the logistic regression model: {e}")
        raise

def evaluate_model(model:LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate the model using various metrics including 'Specificity', 'PPV', and 'NPV'.
    """
    try:
        logging.info("Evaluating the model.")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculating specificity, PPV and NPV
        tn, fp, fn, tp = np.array(cm).ravel()
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Specificity': specificity,
            'F1 Score': f1_score(y_test, y_pred),
            'PPV': ppv,
            'NPV': npv,
            'ROC AUC': roc_auc_score(y_test, y_prob),
            'Confusion Matrix': str(cm)
        }
    
        return pd.DataFrame([metrics])
    except Exception as e:
        logging.error(f"An error occurred while evaluating the model: {e}")
        raise


def perform_model_building(df: pd.DataFrame, model_type: str, target_column: str, test_size: float, random_state: int, class_weight=None) -> pd.DataFrame:
    """
    Perform all model building steps and save the evaluation metrics and baselines to separate CSV files in the predefined folder.
    """
    try:
        logging.info(f"Performing all model building steps for {model_type}.")
        
        # Predefined folder for saving reports
        reports_folder = "reports/"
        
        # File paths for saving the metrics and baselines
        metrics_report_path = f"{reports_folder}{model_type}_metrics_report.csv"
        baseline_report_path = f"{reports_folder}baseline_report.csv"
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)
        
        # # Perform baselining
        # baseline_metrics = perform_baselining(y_test)
        # logging.info(f"Baseline Metrics: {baseline_metrics}")
        
        # Train the specified model
        if model_type == "logistic_regression":
            model = train_logistic_regression(X_train, y_train, random_state, class_weight=class_weight)
        # elif model_type == "random_forest":
        #     model = train_random_forest(X_train, y_train, random_state, class_weight=class_weight)
        # elif model_type == "stochastic_gradient_descent":
        #     model = train_stochastic_gradient_descent(X_train, y_train, random_state, class_weight=class_weight)
        # elif model_type == "support_vector_machine":
        #     model = train_support_vector_machine(X_train, y_train, random_state, class_weight=class_weight)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate the model
        evaluation_metrics = evaluate_model(model, X_test, y_test)
        
        # Save the evaluation metrics and baselines to separate CSV files
        evaluation_metrics.to_csv(metrics_report_path, index=False)
        # baseline_metrics.to_csv(baseline_report_path, index=False)
        
        return evaluation_metrics #, baseline_metrics
    except Exception as e:
        logging.error(f"An error occurred during model building: {e}")
        raise

