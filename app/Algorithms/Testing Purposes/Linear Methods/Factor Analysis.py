import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("training.csv")

X = data.drop(columns=["ONCOGENIC"])
y = data['ONCOGENIC']

fa = FactorAnalysis(n_components=3)

X_fa = fa.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_fa, y, test_size=0.2, random_state=42)

lr = LogisticRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Calculate the training and testing accuracy of the model
training_accuracy = accuracy_score(y_train, y_train_pred)
testing_accuracy = accuracy_score(y_test, y_test_pred)

print('Model: Factor Analysis with Logistic Regression')
print('Training Accuracy:', training_accuracy)
print('Testing Accuracy:', testing_accuracy)
