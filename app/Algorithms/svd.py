import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def eval_svd(path):
    # Load the dataset
    df = pd.read_csv(path)

    # Extract the features and target variable
    X = df.drop('ONCOGENIC', axis=1)
    y = df['ONCOGENIC']

    svd = TruncatedSVD(n_components=2)
    X_svd = svd.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.2, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Model Name: SVD + Logistic Regression")
    print("Accuracy:", train_accuracy)
