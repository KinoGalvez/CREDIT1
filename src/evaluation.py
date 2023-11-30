# evaluation.py`: código para evaluar el modelo utilizando los datos de prueba de la carpeta `data/test` y generar métricas de evaluación.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve,auc
import warnings
warnings.filterwarnings("ignore")
import pickle

X = pd.read_csv('../data/processed/X_balmix_cat_mean.csv')
y = pd.read_csv('../data/processed/y_balmix_cat_mean.csv')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

train = pd.concat([X_train, pd.DataFrame(y_train, columns=['loan_status'])], axis=1)

train.to_csv('../data/train/train_balmix_cat_mean.csv', index=False)
X_test.to_csv('../data/test/test_balmix_cat_mean.csv', index=False)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scal = scaler.fit_transform(X_train)
X_test_scal = scaler.transform(X_test)

gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)

parameters = {"n_estimators":[50,100,150],
              "max_depth": [2,3,4,5],
              "max_features": [2,3,4],
              "learning_rate":[0.01,0.1,0.5]}

gbc_gs = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, scoring="roc_auc", n_jobs=-1, verbose=2)

gbc_gs.fit(X_train_scal, y_train)

print(gbc_gs.best_estimator_)
print(gbc_gs.best_params_)
print(gbc_gs.best_score_)

final_gbc = gbc_gs.best_estimator_
final_gbc.fit(X_train_scal, y_train)
y_pred = final_gbc.predict(X_test_scal)

print("accuracy_score", accuracy_score(y_test, y_pred))
print("recall_score", recall_score(y_test, y_pred))
print("precision_score", precision_score(y_test, y_pred))
print("roc_auc_score", roc_auc_score(y_test, y_pred))
print("f1_score", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

