# %%
import pandas as pd
import seaborn as sns
# from src.data.make_dataset import main
from src.models.train_model import split_data, train_logistic_regression, train_random_forest, train_stochastic_gradient_descent, train_support_vector_machine, evaluate_model
from src.exploration.explore_data import check_and_create_directory, exploratory_data_analysis
from src.features import build_features  # empty
from src.preprocessing.preprocess_data import preprocess_data
from src.visualization.visualize import plot_roc_curve
from src.data_load import load_data


# from imblearn.under_sampling import RandomUnderSampler

# %%
'''''
Columns	      Description

age	          Age in years
ed	          Level of education (1=Did not complete high school, 2=High school degree, 3=Some college, 4= college degree, 5=Post-undergraduate degree)
employ	      Years with current employer
address	      Years at current address
income	      Household income in thousands
debtinc	      Debt to income ratio (x100)
creddebt	  Credit card debt in thousands
othdebt	      Other debt in thousands
default	      Previously defaulted (1= defaulted, 0=Never defaulted)

'''''
sns.set_theme()


def check_and_create_directory(directory_name):
    """Check if a directory exists, if not, create it."""
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)


def load_data(file_path):
    """Load data from CSV file and return a pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Error: File {file_path} not found.")
        return None


def preprocess_data(df):
    """Preprocess the dataframe: convert columns to appropriate types, handle missing values etc."""
    for col in ['ed', 'default']:
        df[col] = df[col].astype('category')
    return df


def exploratory_data_analysis(df, directory_name="images"):
    """Perform exploratory data analysis and save plots to the specified directory."""
    check_and_create_directory(directory_name)

    # Box Plots
    variables = ["age", "ed", "employ", "address",
                 "debtinc", "creddebt", "othdebt"]
    for y in variables:
        if y != "ed":
            sns.boxplot(data=df, x="default", y=y)
            plt.savefig(f"{directory_name}/boxplot_{y}.png")
            plt.clf()

    # Histograms
    for x in variables:
        sns.displot(data=df, x=x, hue='default')
        plt.savefig(f"{directory_name}/histhue_{x}.png")
        plt.clf()


def split_data(df, test_size=0.2, random_state=44):
    """Split data into training and testing sets."""
    X = df[df.columns.drop('default')].values
    y = df["default"].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(model, X_train, y_train):
    """Train a model using the provided training data."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, report


def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()


def perform_undersampling(X_train, y_train, random_state=44):
    RUS = RandomUnderSampler(random_state=random_state)
    X_train_rus, y_train_rus = RUS.fit_resample(X_train, y_train)
    return X_train_rus, y_train_rus
# %%


def main():
    # Load and preprocess data
    df = load_data('bankloan.csv')
    if df is None:  # Check if data loaded successfully
        return

    df = preprocess_data(df)

    # Exploratory Data Analysis
    exploratory_data_analysis(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and evaluate models
    models = {
        "Logistic Regression": LogisticRegression(random_state=44),
        "Random Forest": RandomForestClassifier(random_state=44),
        "Stochastic Gradient Descent": SGDClassifier(random_state=42),
        "Support Vector Machine": SVC(kernel='linear')
    }

    trained_models = {}
    for model_name, model in models.items():
        trained_model = perform_model_building(model, X_train, y_train)
        trained_models[model_name] = trained_model

    # Evaluate models
    for model_name, trained_model in trained_models.items():
        accuracy, conf_matrix, report = evaluate_model(
            trained_model, X_test, y_test)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)
        print("-" * 50)

        if hasattr(trained_model, "predict_proba"):
            plot_roc_curve(trained_model, X_test, y_test)

        print("-" * 50)

    # Apply undersampling

    X_train_rus, y_train_rus = perform_undersampling(X_train, y_train)

    # Retrain and re-evaluate models on undersampled data
    trained_models_rus = {}
    for model_name, model in models.items():
        trained_model_rus = train_model(model, X_train_rus, y_train_rus)
        trained_models_rus[model_name] = trained_model_rus

        # Re-evaluate models
    for model_name, trained_model_rus in trained_models_rus.items():
        accuracy, conf_matrix, report = evaluate_model(
            trained_model_rus, X_test, y_test)
        print(f"Model: {model_name} (After Undersampling)")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)

        # Plot ROC curve for models that support predict_proba
        if hasattr(trained_model_rus, "predict_proba"):
            plot_roc_curve(trained_model_rus, X_test, y_test)


main()


# %%

# %%
