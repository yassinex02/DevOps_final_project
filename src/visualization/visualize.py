import matplotlib.pyplot as plt
from sklearn import metrics

# current_script_directory = Path(__file__).parent
# root_directory = current_script_directory.parent
# config_path = root_directory / "config.yaml"

# with open(config_path, 'r') as config_file:
#     config = yaml.safe_load(config_file)

# # Construct the absolute path to the processed CSV file
# processed_csv_path = root_directory / Path(config['data']['processed_filepath'])


def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()
