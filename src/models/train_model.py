import argparse
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import wandb
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# Initialize logging
logging.basicConfig(level=logging.INFO)



def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame]:
    """
    Split the data into training and test sets.
    """
    try:
        logging.info("Splitting the data into training and test sets.")
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"An error occurred while splitting the data: {e}")
        raise


def random_undersampling(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> Tuple[pd.DataFrame]:
    """
    Perform random undersampling on the training data.
    """
    try:
        logging.info("Random undersampling.")
        rus = RandomUnderSampler(random_state=random_state)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        return X_train_rus, y_train_rus
    except Exception as e:
        logging.error(
            "An error occurred while random undersampling the data: %s", e)
        raise


def perform_baselining(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Perform baselining using both Majority Class and Stratified Random Guessing.
    """
    try:
        logging.info("Performing baselining using DummyClassifier.")
        baselines = {}
        
        # Dummy feature array for baselining
        dummy_X = np.zeros((len(y_test), 1))

        # Majority Class (Zero-Rule Algorithm)
        majority_baseline = DummyClassifier(strategy='most_frequent')
        majority_baseline.fit(dummy_X, y_test)
        y_pred_majority = majority_baseline.predict(dummy_X)
        baselines['Majority Class'] = accuracy_score(y_test, y_pred_majority)
        
        # Stratified Random Guessing
        stratified_baseline = DummyClassifier(strategy='stratified')
        stratified_baseline.fit(dummy_X, y_test)
        y_pred_stratified = stratified_baseline.predict(dummy_X)
        baselines['Stratified Random'] = accuracy_score(y_test, y_pred_stratified)
        
        return pd.DataFrame([baselines])
    except Exception as e:
        logging.error(f"An error occurred while performing baselining: {e}")
        raise


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int, class_weight=None
) -> LogisticRegression:
    """Train a logistic regression model."""
    try:
        logging.info("Training a logistic regression model.")
        lr_model = LogisticRegression(
            random_state=random_state, class_weight=class_weight
        )
        lr_model.fit(X_train, y_train)
        return lr_model
    except Exception as e:
        logging.error(
            f"An error occurred while training the logistic regression model: {e}"
        )
        raise


def evaluate_model(
    model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
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
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Specificity": specificity,
            "F1 Score": f1_score(y_test, y_pred),
            "PPV": ppv,
            "NPV": npv,
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "Confusion Matrix": str(cm),
        }

        return pd.DataFrame([metrics])
    except Exception as e:
        logging.error(f"An error occurred while evaluating the model: {e}")
        raise


def plot_roc_curve(model, X_test, y_test, report_folder):
    """
    Plot the ROC curve for the given model and test data.

    Parameters:
    model (Model): The trained model.
    X_test (DataFrame): The test features.
    y_test (Series): The true labels for the test data.
    report_folder (str): The folder where to save the plot.
    model (str): The type of the model, used for naming the plot file.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model}")
    plt.legend(loc=4)

    # Save the plot to a file
    plt.savefig(f"{report_folder}{model}_roc_curve.png")
    plt.close()


def save_model(model, filename: str):
    """
    Save the trained model to disk.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained machine learning model.
    filename (str): The path to the file where the model should be saved.
    """
    try:
        dump(model, filename)
        logging.info(f"Model saved to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")
        raise


def perform_model_building(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    class_weight=None,
) -> pd.DataFrame:
    """
    Perform all model building steps and save the evaluation metrics and baselines to separate CSV files in the predefined folder.
    """
    try:
        logging.info(f"Performing all model building steps.")

        # Predefined folder for saving reports
        reports_folder = "reports/"

        # File paths for saving the metrics and baselines
        metrics_report_path = f"{reports_folder}_metrics_report.csv"
        baseline_report_path = f"{reports_folder}_baseline_report.csv"

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df, target_column, test_size, random_state
        )

        # Perform random undersampling
        X_train_rus, y_train_rus = random_undersampling(
            X_train, y_train, random_state)

        # # Perform baselining
        baseline_metrics = perform_baselining(X_test,y_test)
        logging.info(f"Baseline Metrics: {baseline_metrics}")

        # Train logistic regression model
        model = train_logistic_regression(X_train_rus, y_train_rus, random_state, class_weight=class_weight)

        # Save the trained model
        model_filename = f"{reports_folder}Logistic_Regression_model.joblib"
        save_model(model, model_filename)

        # Evaluate the model
        evaluation_metrics = evaluate_model(model, X_test, y_test)

        # Plot and save the ROC curve
        plot_roc_curve(model, X_test, y_test, reports_folder)

        # Save the evaluation metrics and baselines to separate CSV files
        evaluation_metrics.to_csv(metrics_report_path, index=False)
        baseline_metrics.to_csv(baseline_report_path, index=False)

        return evaluation_metrics  # , baseline_metrics
    except Exception as e:
        logging.error(f"An error occurred during model building: {e}")
        raise


def main(args):
    """
    The main entry point of the application that performs data splitting,
    model training, and evaluation based on the provided command line arguments.
    """
    wandb.init(job_type="model_training")

    # Path to the input artifact
    input_artifact = wandb.use_artifact(args.input_artifact)
    input_artifact_path = input_artifact.file()

    try:
        # Load input artifact (cleaned data)
        df = pd.read_csv(input_artifact_path)

        # Perform model building
        evaluation_metrics = perform_model_building(
            df=df,
            target_column=args.target_column,
            test_size=args.test_size,
            random_state=args.random_state,
            class_weight=args.class_weight,
        )

        # Output the evaluation metrics
        print(evaluation_metrics)

        # Save and log the model as an artifact
        model_path = "reports/Logistic_Regression_model.joblib"
        model_artifact = wandb.Artifact(
            args.output_artifact_name, 
            type="model",
            description="Trained model artifact"
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        # Save and log all files in the reports folder as an artifact
        reports_artifact = wandb.Artifact(
            "evaluation_metrics", 
            type="metrics",
            description="Evaluation metrics of the model"
        )
        reports_artifact.add_dir("reports/")
        wandb.log_artifact(reports_artifact)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Model Building and Evaluation")
    
    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Name for the input artifact"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of the test set."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="The random state for reproducibility."
    )
    parser.add_argument(
        "--class_weight",
        default=None,
        help="Class weights for imbalanced datasets."
    )
    parser.add_argument(
        "--output_artifact_name",
        type=str,
        required=True,
        help="Name for the output artifact"
    )
    parser.add_argument(
        "--output_artifact_type",
        type=str,
        required=True,
        help="Type of the output artifact"
    )
    parser.add_argument(
        "--output_artifact_description",
        type=str,
        help="Description for the output artifact"
    )
    
    args = parser.parse_args()
    main(args)
