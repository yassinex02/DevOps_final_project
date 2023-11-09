
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)


def run_shap_analysis(model: LogisticRegression, X_test: pd.DataFrame, instance_index: int):
    """
    Runs SHAP analysis on the model for a specific instance in the test set and saves the plots to disk.
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
        summary_plot_path = os.path.join("reports", "shap_summary_plot.png")
        shap.summary_plot(shap_values, X_test)
        plt.savefig(summary_plot_path)
        plt.close()

        # Generate and save force plot for a specific instance
        force_plot_path = os.path.join(
            "reports", f"shap_force_plot_{instance_index}.png")
        shap.force_plot(explainer.expected_value[0], shap_values.values[instance_index,
                        :], X_test.iloc[instance_index, :], matplotlib=True, show=False)
        plt.savefig(force_plot_path)
        plt.close()

    except Exception as e:
        logging.error(f"An error occurred while running SHAP analysis: {e}")


def run_lime_analysis(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, instance_index: int):
    """
    Runs LIME analysis on the model for a specific instance in the test set and saves the plot to disk.
    """
    try:
        logging.info("Running LIME analysis.")

        # Create LIME explainer object
        explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(),
                                                           training_labels=y_test,
                                                           feature_names=X_test.columns,
                                                           class_names=[
                                                               "0", "1"],
                                                           mode='classification')

        # Generate LIME explanation for a specific instance
        exp = explainer.explain_instance(
            X_test.iloc[instance_index].to_numpy(), model.predict_proba)

        # Save the LIME plot
        lime_plot_path = os.path.join(
            "reports", f"lime_plot_{instance_index}.png")
        exp.save_to_file(lime_plot_path)

    except Exception as e:
        logging.error(f"An error occurred while running LIME analysis: {e}")


def extract_feature_importance(model: LogisticRegression, X: pd.DataFrame):
    """
    Extracts feature importance from the model and saves it to a CSV file.
    """
    try:
        logging.info("Extracting feature importance.")

        # Extract feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.coef_[0]
        }).sort_values(by='Importance', ascending=False)

        # Save feature importance to CSV
        feature_importance_path = os.path.join(
            "reports", "feature_importance.csv")
        feature_importance.to_csv(feature_importance_path, index=False)

    except Exception as e:
        logging.error(
            f"An error occurred while extracting feature importance: {e}")
