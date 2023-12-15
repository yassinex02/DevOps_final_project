import ast
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import wandb


def pytest_addoption(parser):
    """
    Add command line options for pytest.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Options:
        --csv (str): Wandb artifact name for the CSV file (required).
        --ref_data (str): File path for the reference data CSV file.
        --ks_alpha (float): Alpha value for the Kolmogorov-Smirnov test (default: 0.05, required).
        --numerical_columns (str): Numerical columns for the Kolmogorov-Smirnov test (required).
        --required_columns (str): Column names and types required in the data (required).
        --known_classes (List[str]): Known classes for the target column (required).
        --missing_values (str): Whether the data has missing values (required).
        --ranges (List[str]): Ranges for the numerical columns (required).
    """
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
    """
    Fixture that loads a CSV file as a pandas DataFrame for data testing.

    Args:
        request: The pytest request object.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Yields:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.

    """
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
    """
    Fixture that loads reference data from a CSV file.

    Args:
        request: The pytest request object.

    Returns:
        pd.DataFrame: The loaded reference data.

    Raises:
        FileNotFoundError: If the reference data file is not found at the provided path.
    """
    ref_data_path = request.config.getoption("--ref_data")
    if not os.path.exists(ref_data_path):
        raise FileNotFoundError(
            f"Reference data file not found at the provided path: {ref_data_path}")

    df = pd.read_csv(ref_data_path)
    return df


@pytest.fixture(scope='session')
def ks_alpha(request):
    """
    Fixture that returns the value of the --ks_alpha command line option.

    Args:
        request: The pytest request object.

    Returns:
        The value of the --ks_alpha command line option.
    """
    return request.config.getoption("--ks_alpha")


@pytest.fixture(scope='session')
def numerical_columns(request):
    """
    Fixture that returns a list of numerical columns.

    Args:
        request: The pytest request object.

    Returns:
        list: A list of numerical columns.

    Raises:
        pytest.fail: If no --numerical_columns option is provided.
    """
    numerical_columns = request.config.getoption("--numerical_columns")
    if not numerical_columns:
        pytest.fail("No --numerical_columns provided")
    return numerical_columns.split(',')


@pytest.fixture(scope='session')
def required_columns(request):
    """
    Fixture that returns the required columns for data checks.

    Args:
        request: The pytest request object.

    Returns:
        A list of required columns.

    Raises:
        pytest.fail: If no --required_columns option is provided.
    """
    required_columns = request.config.getoption("--required_columns")
    if not required_columns:
        pytest.fail("No --required_columns provided")
    return ast.literal_eval(required_columns)


@pytest.fixture(scope='session')
def known_classes(request):
    """
    Fixture that returns a list of known classes based on the command line argument --known_classes.

    Args:
        request: The pytest request object.

    Returns:
        list: A list of known classes.

    Raises:
        pytest.fail: If no --known_classes argument is provided.
    """
    known_classes = request.config.getoption("--known_classes")
    if not known_classes:
        pytest.fail("No --known_classes provided")
    return ast.literal_eval(known_classes[0])


@pytest.fixture(scope='session')
def missing_values(request):
    """
    Fixture that returns the value of the --missing_values command line option.

    Args:
        request: The pytest request object.

    Returns:
        The value of the --missing_values command line option.
    """
    missing_values = request.config.getoption("--missing_values")


@pytest.fixture(scope='session')
def ranges(request):
    """
    Fixture to parse and return the ranges provided as command-line arguments.

    Args:
        request: The pytest request object.

    Returns:
        dict: A dictionary containing the parsed ranges.

    Raises:
        pytest.fail: If no --ranges argument is provided.
    """
    ranges = request.config.getoption("--ranges")
    ranges_str = ranges[0]
    print(ranges)
    if not ranges:
        pytest.fail("No --ranges provided")
    return ast.literal_eval(ranges_str)
