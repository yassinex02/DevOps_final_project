BANK LOAN MLOPS
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

## Custom Modules
In the packages folder, you can add your custom module. For example, there is a custom transformer module
in the packages folder. It is in a folder called transformer with an __init__.py and a file called transformer.py
The conda.yaml will install every custom module in packages with the line "- -e ./packages" seen under pip.
With this you can easily import your own custom module to be used in this project if you wish.

## Project Structure
------------

opiod_analysis/

        .
    ├── .gitignore
    ├── conda.yml
    ├── config.yaml
    ├── LICENSE
    ├── main.py
    ├── Mlproject
    ├── README.md
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
    ├── notebooks/
    │   ├── .gitkeep
    │   ├── ML Project.ipynb
    │   └── ML Project_WandB.ipynb
    ├── packages/
    │   ├── transformer/
    │   │   ├── __init__.py
    │   │   ├── transformer.py
    │   ├── setup.py
    ├── references/
    │   └── .gitkeep
    ├── reports/
    │   ├── .gitkeep
    │   └── figures/
    │       └── .gitkeep
    ├── src/
    │   ├── __init__.py
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
    │   │   ├── __init__.py
    │   ├── model_tests/
    │   │   ├── conda.yaml
    │   │   ├── Mlproject
    │   │   ├── model_testing.py
    │   │   ├── __init__.py        
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
3. Example of running entire pipeline: 
```
mlflow run . -P steps="[loader,exploration,preprocessing,data_check,data_split,model_building,interpretability,model_testing]
```

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