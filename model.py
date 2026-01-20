# checking virtual environment activity
# import sys
# print(sys.base_prefix != sys.prefix)


import pandas as pd
import numpy as np

#  sklearn.preprocessing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression Model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.ensemble import (
    VotingRegressor,
    StackingRegressor,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrices

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint

import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("HR-Employee-Attrition.csv")
print("Dataset shape : ", df.shape, "\n")
df.head(10)

# Divide data according to numerical and categorical Features
X = df.drop(columns="Attrition")
y = df["Attrition"]

num_features = X.select_dtypes(include=["int64","float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns


if "Over18" in cat_features:
    df.drop(columns="Over18", inplace=True)

# Here we apply binary encoding for binary features
df["Attrition"] = df["Attrition"].replace({"Yes": 1, "No": 0})
# df["OverTime"] = df["OverTime"].replace({"Yes": 1, "No": 0})
# df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

onehotencoding_col = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus"
]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

# nuerical Pipeline
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# creating Categorical Pipeline
binary_features = ["OverTime","Gender"]

binary_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")) , # binary encoding Yes/No => 1/0 done before entering pipeline
        ("encoder", OrdinalEncoder())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]
)

# combine numerical and categorical pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("binary", binary_pipeline, binary_features),
        ("cat", cat_pipeline, onehotencoding_col)
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=93))   # here we use a random model for complete the pipeline
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,average="weighted")
rec = recall_score(y_test, y_pred,average="weighted")
f1 = f1_score(y_test, y_pred,average="weighted")

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("roc auc score",roc_auc)
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import pickle

file_name = "model.pkl"

with open(file_name, "wb") as file:
    pickle.dump(model, file)

with open("model.pkl","rb") as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,average="weighted")
rec = recall_score(y_test, y_pred,average="weighted")
f1 = f1_score(y_test, y_pred,average="weighted")

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)


print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
print(classification_report(y_test, y_pred))