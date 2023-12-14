"""
This script tests the trained logistic regression model (or a similar model) against a test dataset. 
It calculates various performance metrics and logs them using wandb and MLflow.
"""

import argparse
import logging
import numpy as np
import mlflow
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transformer.transformer import FactorizeTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def evaluate_model(
    y_test: pd.Series, y_pred: pd.DataFrame, y_prob: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the performance metrics of the model and log them using wandb and MLflow.

    Args:
        y_test: The test labels.
        y_pred: The predicted labels.
        y_prob: The predicted probabilities.
    """

    try:
        logging.info("Evaluating the model.")
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
            "Confusion Matrix": cm,
        }

        return pd.DataFrame([metrics])
    except Exception as e:
        logging.error("An error occurred while evaluating the model: %s", e)
        raise


def main(args):
    """
    The main function for testing the model.

    Args:
    args: Command-line arguments.
    """
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Fetching the model and test data.")

    # Downloading the model
    model_local_path = run.use_artifact(args.model_artifact).download()

    X_test_artifact = wandb.use_artifact(args.X_test_artifact).file()
    X_test = pd.read_csv(X_test_artifact)

    y_test_artifact = wandb.use_artifact(args.y_test_artifact).file()
    y_test = pd.read_csv(y_test_artifact)

    try:
        logger.info("Loading model and performing inference on test set")

        model = mlflow.sklearn.load_model(model_local_path)
        y_pred = model.predict(X_test)

        logger.info("Calculating performance metrics")

        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)

        # Extract the confusion matrix from the dataframe
        cm = metrics['Confusion Matrix'].values[0]

        # Log confusion matrix as an image in wandb
        wandb.log(
            {"confusion_matrix": wandb.sklearn.plot_confusion_matrix(y_test, y_pred, cm)})

        logger.info(f"Performance metrics: {metrics}")

        # Logging metrics to wandb
        wandb.log(metrics.to_dict(orient='records')[0])

    except Exception as e:
        logger.error("Error in model testing: %s", e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the trained model against the test dataset")

    parser.add_argument("--model_artifact", type=str, required=True,
                        help="The MLflow model artifact to test")
    parser.add_argument("--X_test_artifact", type=str, required=True,
                        help="The test features artifact")
    parser.add_argument("--y_test_artifact", type=str, required=True,
                        help="The test labels artifact")

    args = parser.parse_args()
    main(args)
