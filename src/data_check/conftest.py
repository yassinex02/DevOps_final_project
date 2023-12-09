import os
import pytest
import pandas as pd
import wandb
import shutil
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", required=True,
                     help="Wandb artifact name for the CSV file")
    parser.addoption("--ref_data", action="store",
                     help="File path for the reference data CSV file")
    parser.addoption("--ks_alpha", action="store", type=float, default=0.05, required=True,
                     help="Alpha value for the Kolmogorov-Smirnov test")
    parser.addoption("--column_presence_and_type", action="store", nargs="+", required=True,
                        help="Column names and types required in the data")
    parser.addoption("--known_classes", action="store", nargs="+", required=True,
                        help="Known classes for the target column")
    parser.addoption("--missing_values", action="store", required=True,
                        help="Whether the data has missing values")
    parser.addoption("--ranges", action="store", nargs="+", required=True,
                        help="Ranges for the numerical columns")
    


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    artifact_name = request.config.getoption("--csv") 
    artifact = run.use_artifact(artifact_name)
    data_path = artifact.file()
    df = pd.read_csv(data_path)
    yield df
    dir_to_remove = Path(data_path).parents[1]
    if dir_to_remove.exists():
        shutil.rmtree(dir_to_remove)



@pytest.fixture(scope='session')
def ref_data(request):
    ref_data_path = request.config.getoption("--ref_data")
    if not os.path.exists(ref_data_path):
        raise FileNotFoundError(
            f"Reference data file not found at the provided path: {ref_data_path}")

    df = pd.read_csv(ref_data_path)
    return df


@pytest.fixture(scope='session')
def ks_alpha(request):
    return request.config.getoption("--ks_alpha")



@pytest.fixture(scope='session')
def required_columns(request):
    required_columns=request.config.getoption("--column_presence_and_type")
    if not required_columns:
        pytest.fail("No --column_presence_and_type provided")


@pytest.fixture(scope='session')
def known_classes(request):
    known_classes=request.config.getoption("--known_classes")
    if not known_classes:
        pytest.fail("No --known_classes provided")


@pytest.fixture(scope='session')
def missing_values(request):
    missing_values=request.config.getoption("--missing_values")



@pytest.fixture(scope='session')
def ranges(request):
    ranges=request.config.getoption("--ranges")
    if not ranges:
        pytest.fail("No --ranges provided")