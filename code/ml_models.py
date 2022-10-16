import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classification_res(features, labels):
    res_array = np.zeros(shape=(len(features), 6, 3))
    for i, data in enumerate(features):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)
        for j, algo in enumerate(
                [LogisticRegression(), SVC(gamma='auto'), KNeighborsClassifier(n_neighbors=3), GaussianNB(),
                 DecisionTreeClassifier(random_state=0, max_depth=4),
                 RandomForestClassifier(max_depth=10, random_state=0, n_estimators=50)]):
            pipe = make_pipeline(StandardScaler(), algo)
            pipe.fit(X_train, y_train)
            prec, recall, f1, _ = precision_recall_fscore_support(y_test, pipe.predict(X_test), average='binary')
            res_array[i][j][0] = prec
            res_array[i][j][1] = recall
            res_array[i][j][2] = f1

    return res_array


def get_regression_res(features, labels):
    res_array = np.zeros(shape=(len(features), 7, 2))
    for i, data in enumerate(features):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)
        for j, algo in enumerate(
                [linear_model.LinearRegression(), linear_model.BayesianRidge(compute_score=True, n_iter=300),
                 ElasticNet(alpha=0.0, random_state=0, max_iter=1000), KNeighborsRegressor(n_neighbors=2),
                 RandomForestRegressor(max_depth=8, random_state=0, n_estimators=100),
                 make_pipeline(StandardScaler(), SVR(epsilon=0.2, degree=3))]):
            algo.fit(X_train, y_train)
            pred = algo.predict(X_test)
            res_array[i][j][0] = mean_absolute_error(y_test, pred)
            res_array[i][j][1] = np.sqrt(mean_squared_error(y_test, pred))

        X_train_ = np.array(X_train) ** 2
        X_test_ = np.array(X_test) ** 2
        algo = linear_model.BayesianRidge(compute_score=True, n_iter=300)
        algo.fit(X_train_, y_train)
        pred = algo.predict(X_test_)
        res_array[i][6][0] = mean_absolute_error(y_test, pred)
        res_array[i][6][1] = np.sqrt(mean_squared_error(y_test, pred))

    return res_array





