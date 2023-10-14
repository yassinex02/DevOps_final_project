# %%
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
#%%
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
IMAGES = "images"
if not os.path.exists("images"):
    os.mkdir(IMAGES)
#%%
loan = pd.read_csv('bankloan.csv')
#%%
loan.describe() # Summary of Dataset
# %% EXPLORATORY DATA ANALYSIS
loan.info() # We show the content of the dataframe/No missing values

for col in ['ed','default']:
    loan[col] = loan[col].astype('category')

loan.head(10)  # Show the first 10 records of the dataframe

#%%
# Number of people who did and did not defaulted in the past
#To check if our dataset is balanced
print(loan["default"].value_counts())
print(loan["default"].value_counts(normalize=True))
loan.groupby("default").mean()
# %%
# Plotting Box Plots
variables = ["age", "ed", "employ","address","debtinc","creddebt","othdebt"]
for y in variables:
    if y!="ed":
        sns.boxplot(data=loan ,x="default",y=y)
        plt.savefig(f"{IMAGES}/boxplot_{y}.png")
        plt.clf()
# Histogram by Default classes   
for x in variables:
    sns.displot(data=loan ,x=x,hue='default')
    plt.savefig(f"{IMAGES}/histhue_{x}.png")
    plt.clf()

# %% DATA PREPARATION
from sklearn.model_selection import train_test_split
X=loan[loan.columns.drop('default')].values
y=loan["default"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=44,stratify=y)

# %%
y_train
# %%
#%%
#Training 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC  

log_reg = LogisticRegression(solver='lbfgs', random_state=44)
rand_forest = RandomForestClassifier(random_state=44)
sgd = SGDClassifier(random_state=42)
svm = SVC(kernel='linear')

def fit_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model


# %%
'''''
EVALUATION OF THE MODEL
'''''
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import metrics

# def evaluate_model(X_train, y_train, model):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#     return model

fit_model(X_train, y_train, log_reg)

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def new_func(X_test, y_test, model):
    Y_predictions = model.predict(X_test) # Predicting the target variable in the test set
    print(f'{model} = ', accuracy_score(Y_predictions,y_test))


    conf_matrix = confusion_matrix(y_test, Y_predictions)
#print(conf_matrix)
    print(classification_report(y_test, Y_predictions))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    print("\033[1m The result is telling us that we have: ",(conf_matrix[0,0]+conf_matrix[1,1]),"correct predictions\033[1m")
    print("\033[1m The result is telling us that we have: ",(conf_matrix[0,1]+conf_matrix[1,0]),"incorrect predictions\033[1m")
    print("\033[1m We have a total predictions of: ",(conf_matrix.sum()))


    y_pred_proba = log_reg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
#%%
rand_forest = RandomForestClassifier(random_state=44)
rand_forest.fit(X_train, y_train)

Y_predictions = rand_forest.predict(X_test) # Predicting the target variable in the test set
print('Random Forest accuracy = ', accuracy_score(Y_predictions,y_test))

conf_matrix = confusion_matrix(y_test, Y_predictions)
# print(conf_matrix)
print(classification_report(y_test, Y_predictions))
# %%
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
Y_predictions = sgd.predict(X_test) # Predicting the target variable in the test set

print('S Gradient Descent = ', accuracy_score(Y_predictions,y_test))
conf_matrix = confusion_matrix(y_test, Y_predictions)
#print(conf_matrix)
print(classification_report(y_test, Y_predictions))
# %%
from sklearn.svm import SVC  
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
Y_predictions = svm.predict(X_test) # Predicting the target variable in the test set
print('Support Vector Machine = ', accuracy_score(Y_predictions,y_test))

conf_matrix = confusion_matrix(y_test, Y_predictions)
# print(conf_matrix)
print(classification_report(y_test, Y_predictions))
# %% 
"""""
UNDERSAMPLING
"""""
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

RUS = RandomUnderSampler(random_state=44)

#Fit the RUS
X_train_rus, y_train_rus= RUS.fit_resample(X_train, y_train)

# Check the number of records after under sampling
print(sorted(Counter(y_train_rus).items()))
# %%
from sklearn.model_selection import cross_val_score

log_reg = LogisticRegression(solver='lbfgs', random_state=44)
log_reg.fit(X_train_rus, y_train_rus)

Y_predictions = log_reg.predict(X_test) # Predicting the target variable in the test set
print('Logistic Regression accuracy = ', accuracy_score(Y_predictions,y_test))

conf_matrix = confusion_matrix(y_test, Y_predictions)
#print(conf_matrix)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print("\033[1m The result is telling us that we have: ",(conf_matrix[0,0]+conf_matrix[1,1]),"correct predictions\033[1m")
print("\033[1m The result is telling us that we have: ",(conf_matrix[0,1]+conf_matrix[1,0]),"incorrect predictions\033[1m")
print("\033[1m We have a total predictions of: ",(conf_matrix.sum()))

print(classification_report(y_test, Y_predictions))
#%%
log_reg_eval=cross_val_score(estimator=log_reg,X=X_train_rus,y=y_train_rus, cv=10)
print("Score Using CV: ", log_reg_eval.mean())
#%%
#Metrics
y_pred_proba = log_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#ROC CURVE
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
#%%
rand_forest = RandomForestClassifier(random_state=44)
rand_forest.fit(X_train_rus, y_train_rus)

Y_predictions = rand_forest.predict(X_test) # Predicting the target variable in the test set
print('Random Forest accuracy = ', accuracy_score(Y_predictions,y_test))

conf_matrix = confusion_matrix(y_test, Y_predictions)
#print(conf_matrix)
print(classification_report(y_test, Y_predictions))

# %%
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train_rus, y_train_rus)
Y_predictions = sgd.predict(X_test) # Predicting the target variable in the test set

print('S Gradient Descent = ', accuracy_score(Y_predictions,y_test))
conf_matrix = confusion_matrix(y_test, Y_predictions)
print(conf_matrix)
print(classification_report(y_test, Y_predictions))
#%%
from sklearn.svm import SVC  
svm = SVC(kernel='linear')
svm.fit(X_train_rus, y_train_rus)
Y_predictions = svm.predict(X_test) # Predicting the target variable in the test set
print('Support Vector Machine = ', accuracy_score(Y_predictions,y_test))

conf_matrix = confusion_matrix(y_test, Y_predictions)
print(conf_matrix)
print(classification_report(y_test, Y_predictions))
# # %%



# %%
