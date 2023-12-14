"""
The script below runs model interpretability analysis on a trained model.
"""
import argparse
import logging
import os

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import wandb
from sklearn.linear_model import LogisticRegression
from transformer.transformer import FactorizeTransformer

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def get_model_from_pipeline(pipeline):
    """
    Extracts the logistic regression model from a scikit-learn pipeline.
    """
    if "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]
    else:
        raise ValueError("Logistic regression model not found in pipeline")


def run_shap_analysis(
    model: LogisticRegression, X_test: pd.DataFrame, instance_index: int
):
    """
    Runs SHAP analysis on the model for a specific instance
    in the test set and saves the plots to disk.
    """
    try:
        logging.info("Running SHAP analysis.")

        # Initialize JavaScript for SHAP
        shap.initjs()

        # Create SHAP explainer object
        explainer = shap.Explainer(model, X_test)

        # Compute SHAP values
        shap_values = explainer(X_test)

        # Generate and save summary plot
        os.makedirs("reports", exist_ok=True)
        summary_plot_path = os.path.join("reports", "shap_summary_plot.png")
        shap.summary_plot(shap_values, X_test)
        plt.savefig(summary_plot_path)
        plt.close()

        # Generate and save force plot for a specific instance
        force_plot_path = os.path.join(
            "reports", f"shap_force_plot_{instance_index}.png"
        )
        shap.force_plot(
            explainer.expected_value[0],
            shap_values.values[instance_index, :],
            X_test.iloc[instance_index, :],
            matplotlib=True,
            show=False,
        )
        plt.savefig(force_plot_path)
        plt.close()

    except Exception as e:
        logging.error("An error occurred while running SHAP analysis: %s", e)


def run_lime_analysis(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    instance_index: int,
):
    """
    Runs LIME analysis on the model for a specific instance
    in the test set and saves the plot to disk.
    """
    try:
        logging.info("Running LIME analysis.")

        # Create LIME explainer object
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test.to_numpy(),
            training_labels=y_test,
            feature_names=X_test.columns,
            class_names=["0", "1"],
            mode="classification",
        )

        # Generate LIME explanation for a specific instance
        exp = explainer.explain_instance(
            X_test.iloc[instance_index].to_numpy(), model.predict_proba
        )

        # Save the LIME plot
        lime_plot_path = os.path.join("reports", f"lime_plot_{instance_index}.png")
        exp.save_to_file(lime_plot_path)

    except Exception as e:
        logging.error("An error occurred while running LIME analysis: %s", e)


def extract_feature_importance(model: LogisticRegression, X: pd.DataFrame):
    """
    Extracts feature importance from the model and saves it to a CSV file.
    """
    try:
        logging.info("Extracting feature importance.")

        # Extract feature importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.coef_[0]}
        ).sort_values(by="Importance", ascending=False)

        # Save feature importance to CSV
        feature_importance_path = os.path.join("reports", "feature_importance.csv")
        feature_importance.to_csv(feature_importance_path, index=False)

    except Exception as e:
        logging.error("An error occurred while extracting feature importance: %s", e)


def main(args):
    """
    The main function for running model interpretability analysis.
    """

    run = wandb.init(job_type="model_interpretability")
    run.config.update(args)

    # Fetching and loading the test data from wandb artifacts
    X_test_artifact = wandb.use_artifact(args.X_test_artifact).file()
    X_test = pd.read_csv(X_test_artifact)

    y_test_artifact = wandb.use_artifact(args.y_test_artifact).file()
    y_test = pd.read_csv(y_test_artifact).squeeze()

    # Fetching and loading the model from wandb artifacts
    model_local_path = run.use_artifact(args.model_artifact).download()
    pipeline = mlflow.sklearn.load_model(model_local_path)
    model = get_model_from_pipeline(pipeline)

    # Run SHAP analysis
    run_shap_analysis(model, X_test, args.instance_index)

    # Run LIME analysis
    run_lime_analysis(model, X_test, y_test, args.instance_index)

    # Extract feature importance
    extract_feature_importance(model, X_test)

    # Log the reports folder as an artifact in wandb
    reports_artifact = wandb.Artifact(
        name=args.reports_artifact,
        type="reports",
        description="Model interpretability reports",
    )
    reports_artifact.add_dir("reports")
    run.log_artifact(reports_artifact)

    logger.info("Model interpretability analysis completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model interpretability analysis")

    parser.add_argument(
        "--model_artifact",
        type=str,
        required=True,
        help="The MLflow model artifact to test",
    )
    parser.add_argument(
        "--X_test_artifact", type=str, required=True, help="The test features artifact"
    )
    parser.add_argument(
        "--y_test_artifact", type=str, required=True, help="The test labels artifact"
    )
    parser.add_argument(
        "--instance_index",
        type=int,
        required=True,
        help="The index of the instance to explain",
    )
    parser.add_argument(
        "--reports_artifact",
        type=str,
        required=True,
        help="Name of the reports artifact to log to W&B",
    )

    args = parser.parse_args()
    main(args)
