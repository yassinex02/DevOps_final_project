import argparse
import json
import logging
import os
import shutil
from typing import Tuple
import mlflow
import pandas as pd
import wandb
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformer import FactorizeTransformer


# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def get_preprocessing_pipeline(numerical_cols, factorize_cols):
    if isinstance(numerical_cols, str):
        numerical_cols = [numerical_cols]

    if isinstance(factorize_cols, str):
        factorize_cols = [factorize_cols]

    numerical_transformer = StandardScaler()
    factorize_transformer = FactorizeTransformer()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('fact', factorize_transformer, factorize_cols),
        ])

    return preprocessor


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


def train_logistic_regression(
        X_train: pd.DataFrame, y_train: pd.Series, hyperparameters) -> LogisticRegression:

    model = LogisticRegression(**hyperparameters)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    y_test: pd.Series, y_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the model using various metrics including 'Specificity', 'PPV', and 'NPV'.
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
            "ROC AUC": roc_auc_score(y_test, y_pred),
        }

        return pd.DataFrame([metrics])
    except Exception as e:
        logging.error(f"An error occurred while evaluating the model: {e}")
        raise


def main(args):
    run = wandb.init(job_type="train_logistic_regression")
    run.config.update(args)  # Logs all current config to wandb

    # Load configurations from JSON files
    with open(args.hyperparameters) as f:
        hyperparameters = json.load(f)

    try:
        # Fetching and loading the training data from wandb artifacts
        X_train_artifact = wandb.use_artifact(args.X_train_artifact).file()
        X_train = pd.read_csv(X_train_artifact)

        y_train_artifact = wandb.use_artifact(args.y_train_artifact).file()
        y_train = pd.read_csv(y_train_artifact).squeeze()

        # Splitting the training data into smaller training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.val_size, random_state=args.random_seed)

        # Random undersampling
        X_train_resampled, y_train_resampled = random_undersampling(
            X_train, y_train, args.random_seed)

        # Get preprocessing pipeline
        # Split the numerical_cols if it's a single string
        if len(args.numerical_cols) == 1 and ' ' in args.numerical_cols[0]:
            args.numerical_cols = args.numerical_cols[0].split()
        print(args.numerical_cols)
        # Split the factorize_cols if it's a single string
        if len(args.factorize_cols) == 1 and ' ' in args.factorize_cols[0]:
            args.factorize_cols = args.factorize_cols[0].split()
        preprocessing_pipeline = get_preprocessing_pipeline(
            numerical_cols=args.numerical_cols,
            factorize_cols=args.factorize_cols
        )

        # Train logistic regression model
        logistic_model = Pipeline(steps=[
            ('preprocessor', preprocessing_pipeline),
            ('classifier', train_logistic_regression(
                X_train_resampled, y_train_resampled, hyperparameters))
        ])

        # Fit the pipeline to the resampled training data
        logistic_model.fit(X_train_resampled, y_train_resampled)

        # Evaluate the model on the validation set
        y_pred = logistic_model.predict(X_val)
        y_prob = logistic_model.predict_proba(
            X_val)[:, 1]  # Probability estimates
        performance_metrics_df = evaluate_model(y_val, y_pred)

        # Plot ROC curve

        # Convert y_prob to a 2D array for wandb ROC plotting
        y_prob_2d = np.vstack((1 - y_prob, y_prob)).T

        wandb.sklearn.plot_roc(y_val, y_prob_2d, logistic_model.classes_)

        # Plot Precision-Recall curve
        wandb.sklearn.plot_precision_recall(
            y_val, y_prob_2d, logistic_model.classes_)

        # Plotting the confusion matrix
        wandb.sklearn.plot_confusion_matrix(
            y_val, y_pred, logistic_model.classes_)

        logger.info(f"Performance metrics: {performance_metrics_df}")
        # Log the model and metrics to wandb
        wandb.log(
            {'performance_metrics': performance_metrics_df.to_dict(orient='records')[0]})

        # Infer the signature of the model and save the model
        signature = infer_signature(X_val, y_pred)
        model_dir = "logistic_regression_model_dir"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(
            logistic_model,
            model_dir,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[[0]]
        )

        # Log the MLflow model directory as an artifact in wandb
        model_artifact = wandb.Artifact(
            name=args.model_artifact,
            type="model",
            description="Trained logistic regression model with feature engineering"
        )
        model_artifact.add_dir(model_dir)
        artifact = run.log_artifact(model_artifact)

        # Tagging the artifact version as 'prod'
        artifact.wait()
        api = wandb.Api()
        logged_artifact = api.artifact(
            f"{run.entity}/{run.project}/{args.model_artifact}:latest")
        if 'prod' not in logged_artifact.aliases:
            logged_artifact.aliases.append('prod')
            logged_artifact.save()

        logger.info("Model training and evaluation completed.")

    except Exception as e:
        logger.error(f"Error in model training or evaluation: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a logistic regression model")

    parser.add_argument("--X_train_artifact", type=str,
                        required=True, help="W&B artifact name for X_train data")
    parser.add_argument("--y_train_artifact", type=str,
                        required=True, help="W&B artifact name for y_train data")
    parser.add_argument("--val_size", type=float, required=True,
                        help="Size for the validation set split")
    parser.add_argument("--numerical_cols", nargs='+', required=True,
                        help="List of column names to be treated as numerical features")
    parser.add_argument("--factorize_cols", nargs='+',
                        required=True, help="List of column names to factorize")
    parser.add_argument("--hyperparameters", type=str, required=True,
                        help="JSON file with hyperparameters for the logistic regression model")
    parser.add_argument("--model_artifact", type=str,
                        required=True, help="Name of the model to log to W&B")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for data splitting and undersampling")

    args = parser.parse_args()
    main(args)

