opiod_analysis
==============================

## Description
Credit scoring Project to predict whether or not credit extended to an applicant will likely result in profit or losses for the lending institution. The project main is to learn MLOPS best practices.
It also includes fairness and interpretability analysis.

## Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Create a conda environment using the `conda.yml` file with the following command:
    ```
    conda env create -f conda.yml
    ```
4. Activate the conda environment:
    ```
    conda activate devops_project
    ```

## Configuration
The project uses a `config.yaml` file for configuration settings, including data paths, model parameters, and feature engineering settings. Modify this file as needed.

## Project Structure
------------

opiod_analysis/

        .
    ├── .gitignore
    ├── conda.yml
    ├── config.yaml
    ├── dd.py
    ├── LICENSE
    ├── main.py
    ├── Makefile
    ├── Mlproject
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── temp_file.csv
    ├── test_environment.py
    ├── tox.ini
    ├── data/
    │   ├── external/
    │   │   └── .gitkeep
    │   ├── interim/
    │   │   └── .gitkeep
    │   ├── processed/
    │   │   ├── .gitkeep
    │   │   ├── bankloan_processed.csv
    │   │   └── cleaned_data.csv
    │   ├── raw/
    │   │   ├── .gitkeep
    │   │   └── bankloan.csv
    │   └── ref/
    │       └── ref_bankloan.csv
    ├── docs/
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   ├── make.bat
    │   └── Makefile
    ├── models/
    │   └── .gitkeep
    ├── notebooks/
    │   ├── .gitkeep
    │   ├── ML Project.py
    │   ├── ML Project_Refactored.py
    │   └── ML Project_WandB.ipynb
    ├── references/
    │   └── .gitkeep
    ├── reports/
    │   ├── .gitkeep
    │   └── figures/
    │       └── .gitkeep
    ├── src/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── .gitkeep
    │   │   ├── make_dataset.py
    │   │   └── __init__.py
    │   ├── data_check/
    │   │   ├── conda.yml
    │   │   ├── conftest.py
    │   │   ├── Mlproject
    │   │   ├── test_data.py
    │   │   └── __init__.py
    │   ├── data_split/
    │   │   ├── conda.yaml
    │   │   ├── Mlproject
    │   │   ├── split_train_test.py
    │   │   └── __init__.py
    │   ├── exploration/
    │   │   ├── conda.yaml
    │   │   ├── explore_data.py
    │   │   ├── Mlproject
    │   │   └── __init__.py
    │   ├── features/
    │   │   ├── .gitkeep
    │   │   ├── build_features.py
    │   │   └── __init__.py
    │   ├── interpretability/
    │   │   ├── conda.yaml
    │   │   ├── Mlproject
    │   │   ├── model_interpretability.py
    │   │   └── __init__.py
    │   ├── load_data/
    │   │   ├── conda.yml
    │   │   ├── data_load.py
    │   │   ├── Mlproject
    │   │   └── __init__.py
    │   ├── models/
    │   │   ├── conda.yaml
    │   │   ├── Mlproject
    │   │   ├── train_model.py
    │   │   ├── transformer.py
    │   │   ├── __init__.py
    │   │   └── reports/
    │   └── preprocessing/
    │       ├── conda.yml
    │       ├── Mlproject
    │       └── preprocess_data.py
    └── tests/
        ├── __init__.py
        ├── test_data_load.py
        ├── test_explore_data.py
        ├── test_model_interpretability.py
        ├── test_model_test.py
        ├── test_preprocess_data.py
        ├── test_split_train.py
        ├── test_train_model.py
        └── data/
            ├── test_data.csv
            ├── test_data.json
            ├── test_data.txt
            └── test_data.xlsx
        
--------

## Usage
1. Make sure you are in the conda environment.
2. Run your Python scripts or Jupyter Notebooks within this environment to ensure all dependencies are available.

## Testing
Tests can be run using `pytest`. Make sure to install `pytest` if you haven't, and then run the following command from the project directory:
```
pytest tests/
```

## Interpretability and Fairness
- Interpretability is implemented using both SHAP and LIME techniques. Refer to `interpretability.py` for more details.
- Fairness metrics including demographic parity and equalized odds are calculated. Refer to `fairness_analysis.py` for more details.

## License
This project is licensed under the MIT License.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>