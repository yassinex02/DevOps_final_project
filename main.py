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
    'data_check'
    'data_split',
    'model_building',
    'model_testing',
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
                    # "factorize_columns": " ".join(config["preprocess_data"]["factorize_columns"]),
                    # "standardize_columns": " ".join(config["preprocess_data"]["standardize_columns"]),
                    "output_artifact_name": config["preprocess_data"]["output_artifact_name"],
                    "output_artifact_type": config["preprocess_data"]["output_artifact_type"],
                    "output_artifact_description": config["preprocess_data"]["output_artifact_description"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise
    


    if "data_check" in active_steps:
            # Run the Data Quality Checks step
            try:
                _ = mlflow.run(
                    os.path.join(root_path, "src", "data_check"),
                    "main",
                    parameters={
                        "csv": f"{config['preprocess_data']['output_artifact_name']}:latest",
                        "ref_data": config['data_check']['ref_data'],
                        "ks_alpha": config['data_check']['ks_alpha'],
                        "numerical_columns": ",".join(config['data_check']['numerical_columns']),
                        "required_columns": config['data_check']['required_columns'],
                        "known_classes": config['data_check']['known_classes'],
                        "missing_values": config['data_check']['missing_values'],
                        "ranges": config['data_check']['ranges']
                    }
                )
            except Exception as e:
                logger.error("Data Quality Checks MLflow project failed: %s", e)
                raise


    if "data_split" in active_steps:
        # Run the data split step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_split"),
                "main",
                parameters={
                    "input": config['preprocess_data']['output_artifact_name'] + ":latest",
                    "target_column": config["split_data"]["target_column"],
                    "test_size": config["split_data"]["test_size"],
                    "random_state": config["split_data"]["random_state"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise

    if "model_building" in active_steps: 
        # Create a temporary file to store the hyperparameters
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json') as hyperparam_file:

            json.dump(omegaconf.OmegaConf.to_container(
            config["model_building"]["hyperparameters"]), hyperparam_file)

            hyperparam_file_path = hyperparam_file.name

        # Run the model building step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "model_building"),
                "main",
                parameters={
                    "X_train_artifact": config['split_data']['X_train_artifact'] + ":latest",
                    "y_train_artifact": config['split_data']['y_train_artifact'] + ":latest",
                    "val_size": config["model_building"]["val_size"],
                    "numerical_cols": " ".join(config["model_building"]["numerical_cols"]),
                    "factorize_cols": " ".join(config["model_building"]["factorize_cols"]),
                    "hyperparameters": hyperparam_file_path,
                    "model_artifact": config["model_building"]["model_artifact"],
                    "random_seed": config["model_building"]["random_seed"],
                },
            )
        except Exception as e:
                logger.error("MLflow project failed: %s", e)
                raise
        

    if "model_testing" in active_steps:
        # Run the Model Testing step
        try:
            _ = mlflow.run(
                os.path.join(root_path, "src", "model_test"),
                "main",
                parameters={
                    "model_artifact": f"{config['model_building']['model_artifact']}:prod",
                    "X_test_artifact": f"{config['model_testing']['X_test_artifact']}:latest",
                    "y_test_artifact": f"{config['model testing']['y_test_artifact']}:latest",
                }
            )
        except Exception as e:
            logger.error("Model Testing MLflow project failed: %s", e)
            raise

if __name__ == "__main__":
    main()

