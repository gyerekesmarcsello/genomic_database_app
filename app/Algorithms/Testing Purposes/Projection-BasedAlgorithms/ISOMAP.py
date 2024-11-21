import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset using pandas
data = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Instantiate an ISOMAP object
isomap = Isomap(n_neighbors=15, n_components=2)

# Fit the ISOMAP object to the training data and transform the data
X_train_transformed = isomap.fit_transform(X_train)

# Train a logistic regression classifier on the transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_train_transformed, y_train)
print('Training loss: {:.4f}'.format(clf.score(X_train_transformed, y_train)))
# Transform the test data using the fitted ISOMAP object
X_test_transformed = isomap.transform(X_test)

# Predict the labels for the test data
y_pred = clf.predict(X_test_transformed)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
