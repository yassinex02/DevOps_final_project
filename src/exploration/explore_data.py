import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Configure a file handler to log errors only
file_handler = logging.FileHandler('eda_log.log')
file_handler.setLevel(logging.ERROR)  # Log only errors
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure a console handler to display both info and errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Display info and above
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def check_and_create_directory(directory_name):
    """Check if a directory exists, if not, create it."""
    try:
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            logger.info(f"Created directory: {directory_name}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_name}: {str(e)}")


def exploratory_data_analysis(df, directory_name="images"):
    """Perform exploratory data analysis and save plots to the specified directory."""
    check_and_create_directory(directory_name)

    # Box Plots
    variables = ["age", "ed", "employ", "address",
                 "debtinc", "creddebt", "othdebt"]
    for y in variables:
        if y != "ed":
            try:
                sns.boxplot(data=df, x="default", y=y)
                plt.savefig(f"{directory_name}/boxplot_{y}.png")
                plt.clf()
                logger.info(f"Saved boxplot for {y}")
            except Exception as e:
                logger.error(f"Error creating boxplot for {y}: {str(e)}")

    # Histograms
    for x in variables:
        try:
            sns.displot(data=df, x=x, hue='default')
            plt.savefig(f"{directory_name}/histhue_{x}.png")
            plt.clf()
            logger.info(f"Saved histogram for {x}")
        except Exception as e:
            logger.error(f"Error creating histogram for {x}: {str(e)}")
