import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import metrics
# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".."))
base_dir = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "..")

df = pd.read_csv(os.path.join(base_dir, "data", 'BankChurners.csv'))
columns = ["Customer_Age", 'Dependent_count', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
for column_name in columns:
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column_name] < (Q1 - 3 * IQR)) |(df[column_name] > (Q3 + 3 * IQR)))]
df.drop("Credit_Limit", axis = 1, inplace = True)
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer':1, 'Existing Customer':0})
df_dummies = pd.get_dummies(df)
X = df_dummies.drop("Attrition_Flag", axis = 1)
y = df_dummies["Attrition_Flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train - Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

# Test = 0.20
# Validation = 0.20
# Train = 0.60

x_train_shape = X_train.shape
x_val_shape = X_val.shape
x_test_shape = X_test.shape
print("X_train shape = {}\nX_val shape = {}\nX_test shape = {}".format(x_train_shape, x_val_shape, x_test_shape))

def conf_mtrx(y_test, y_pred):
    f, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f",
                ax=ax)
    plt.xlabel("predicted y values")
    plt.ylabel("real y values")
    plt.title("\nConfusion Matrix")
    plt.show()


def rc_recis_scres(y_test, y_pred, algorithm_name):
    from sklearn.metrics import recall_score, precision_score, accuracy_score, \
        f1_score

    rs = recall_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("rs:{}, ps:{}, f1:{}".format(rs, ps, f1))


def ML_Algorithms(model, alg_name, x_trainn, x_testt, y_trainn, y_testt):

    modell = model()
    result_model = modell.fit(x_trainn, y_trainn)
    y_pred = result_model.predict(x_testt)
    conf_mtrx(y_testt, y_pred)
    print("*****", alg_name, " ALGORITHM:")
    rc_recis_scres(y_testt, y_pred, alg_name)


    return model

# ML_Algorithms(RandomForestClassifier, "Random Forest Classifier", X_train, X_val, y_train, y_val)
# ML_Algorithms(DecisionTreeClassifier, "DecisionTreeClassifier", X_train, X_val, y_train, y_val)
# ML_Algorithms(GradientBoostingClassifier, "GradientBoostingClassifier", X_train, X_test, y_train, y_test)
# ML_Algorithms(MLPClassifier, "LGBM Classifier", X_train, X_val, y_train, y_val)
