import os
import pytest
import pandas as pd
import wandb
import shutil
import ast
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", required=True,
                     help="Wandb artifact name for the CSV file")
    parser.addoption("--ref_data", action="store",
                     help="File path for the reference data CSV file")
    parser.addoption("--ks_alpha", action="store", type=float, default=0.05, required=True,
                     help="Alpha value for the Kolmogorov-Smirnov test")
    parser.addoption("--numerical_columns", action="store", required=True,
                    help="Numerical columns for the Kolmogorov-Smirnov test")
    parser.addoption("--required_columns", action="store", required=True,
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
def numerical_columns(request):
    numerical_columns=request.config.getoption("--numerical_columns")
    # # Ensure numerical_columns is a list
    # if not isinstance(numerical_columns, list):
    #     numerical_columns = [numerical_columns]
    # numerical_columns = [word for strings in numerical_columns for word in strings.split()]
    # print(numerical_columns)
    if not numerical_columns:
        pytest.fail("No --numerical_columns provided")
    return numerical_columns.split(',')


@pytest.fixture(scope='session')
def required_columns(request):
    required_columns=request.config.getoption("--required_columns")
    if not required_columns:
        pytest.fail("No --required_columns provided")
    return ast.literal_eval(required_columns)

@pytest.fixture(scope='session')
def known_classes(request):
    known_classes=request.config.getoption("--known_classes")
    if not known_classes:
        pytest.fail("No --known_classes provided")
    return ast.literal_eval(known_classes[0])


@pytest.fixture(scope='session')
def missing_values(request):
    missing_values=request.config.getoption("--missing_values")


@pytest.fixture(scope='session')
def ranges(request):
    ranges=request.config.getoption("--ranges")
    ranges_str = ranges[0]
    print(ranges)
    if not ranges:
        pytest.fail("No --ranges provided")
    return ast.literal_eval(ranges_str)
