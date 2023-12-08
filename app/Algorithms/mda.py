import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


def eval_mda(path):
    # Load the data
    data = pd.read_csv(path)

    X = data.drop('ONCOGENIC', axis=1)
    y = data['ONCOGENIC']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_components = 2 # We only have two classes now
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X_train)

    X_train_gmm = gmm.predict_proba(X_train)
    X_test_gmm = gmm.predict_proba(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_gmm, y_train)

    train_acc = lda.score(X_train_gmm, y_train)

    test_acc = lda.score(X_test_gmm, y_test)

    print("Model Name: Mixture Discriminant Analysis")
    print("Accuracy: ", train_acc)
