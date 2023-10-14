import logging
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# current_script_directory = Path(__file__).parent
# root_directory = current_script_directory.parent
# config_path = root_directory / "config.yaml"

# with open(config_path, 'r') as config_file:
#     config = yaml.safe_load(config_file)

# # Construct the absolute path to the processed CSV file
# processed_csv_path = root_directory / Path(config['data']['processed_filepath'])



def exploratory_data_analysis(df): #TO DO: Deal with the directory location and name later
    """Perform exploratory data analysis and save plots to the specified directory."""
    # check_and_create_directory(directory_name)
    
    # Box Plots
    variables = ["age", "ed", "employ", "address", "debtinc", "creddebt", "othdebt"]
    for y in variables:
        if y != "ed":
            sns.boxplot(data=df, x="default", y=y)
            # plt.savefig(f"{directory_name}/boxplot_{y}.png")
            plt.clf()
            
    # Histograms
    for x in variables:
        sns.displot(data=df, x=x, hue='default')
        # plt.savefig(f"{directory_name}/histhue_{x}.png")
        plt.clf()
