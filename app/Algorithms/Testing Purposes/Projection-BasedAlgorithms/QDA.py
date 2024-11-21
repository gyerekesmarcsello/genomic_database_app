import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset using pandas
data = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=2)

# Instantiate a QDA object
qda = QuadraticDiscriminantAnalysis()

# Fit the QDA object to the training data and print the loss during training
qda.fit(X_train, y_train)
print('Training loss: {:.4f}'.format(qda.score(X_train, y_train)))

# Predict the labels for the test data
y_pred = qda.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
