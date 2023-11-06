import logging
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt




import logging
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(format=config['logging']['format'], level=config['logging']['level'].upper())

def exploratory_data_analysis(df): #TO DO: Deal with the directory location and name later
    """Perform exploratory data analysis and save plots to the specified directory."""
    logging.info('Starting exploratory data analysis')
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
    logging.info('Finished exploratory data analysis')
    