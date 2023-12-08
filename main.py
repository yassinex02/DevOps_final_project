import os
import json
import tempfile
import hydra
import omegaconf
from omegaconf import DictConfig
import logging
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_steps = [
    'loader',
    'exploration',
    'preprocessing',
    'model_building',
]


@hydra.main(config_name="config")
def main(config: DictConfig):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    # active_steps = steps_par.split(",") if steps_par != "all" else _steps
    active_steps = steps_par if steps_par != ["all"] else _steps

     # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    if "loader" in active_steps:
        # Run the data loading step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "load_data"),
                "main",
                parameters={
                    "file_path": config["data_load"]["file_path"],
                    "artifact_name": config["data_load"]["artifact_name"],
                    "sheet_name": config["data_load"]["sheet_name"],
                    "artifact_type": config["data_load"]["artifact_type"],
                    "artifact_description": config["data_load"]["artifact_description"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise
    
    if "exploration" in active_steps:
        # Run the data exploration step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "exploration"),
                "main",
                parameters={
                    "input_artifact": config['data_load']['artifact_name'] + ":latest",
                    "report_title": config["explore_data"]["report_title"],
                    "output_file": config["explore_data"]["output_file"],
                    "expected_columns": " ".join(config["explore_data"]["expected_columns"]),
                    "columns": " ".join(config["explore_data"]["columns"]),
                    "output_artifact_name": config["explore_data"]["output_artifact_name"],
                    "output_artifact_type": config["explore_data"]["output_artifact_type"],
                    "output_artifact_description": config["explore_data"]["output_artifact_description"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise
        
    if "preprocessing" in active_steps:
        # Run the data preprocessing step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "preprocessing"),
                "main",
                parameters={
                    "input_artifact": config['data_load']['artifact_name'] + ":latest",
                    "drop_columns": config["preprocess_data"]["drop_columns"],
                    "factorize_columns": " ".join(config["preprocess_data"]["factorize_columns"]),
                    "standardize_columns": " ".join(config["preprocess_data"]["standardize_columns"]),
                    "output_artifact_name": config["preprocess_data"]["output_artifact_name"],
                    "output_artifact_type": config["preprocess_data"]["output_artifact_type"],
                    "output_artifact_description": config["preprocess_data"]["output_artifact_description"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise
        
    if "model_building" in active_steps:
        # Run the model building step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "models"),
                "main",
                parameters={
                    "input_artifact": config['preprocess_data']['output_artifact_name'] + ":latest",
                    "target_column": config["train_model"]["target_column"],
                    "test_size": config["train_model"]["test_size"],
                    "random_state": config["train_model"]["random_state"],
                    "class_weight": config["train_model"]["class_weight"],
                    "output_artifact_name": config["train_model"]["output_artifact_name"],
                    "output_artifact_type": config["train_model"]["output_artifact_type"],
                    "output_artifact_description": config["train_model"]["output_artifact_description"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise

if __name__ == "__main__":
    main()