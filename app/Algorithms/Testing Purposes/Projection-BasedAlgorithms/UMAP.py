import pandas as pd
import numpy as np
import umap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset using pandas
data = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Instantiate a UMAP object
umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)

# Fit the UMAP object to the training data and transform the data
X_train_transformed = umap_model.fit_transform(X_train)

# Train a logistic regression classifier on the transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_train_transformed, y_train)
print('Training loss: {:.4f}'.format(clf.score(X_train_transformed, y_train)))
# Transform the test data using the fitted UMAP object
X_test_transformed = umap_model.transform(X_test)

# Predict the labels for the test data
y_pred = clf.predict(X_test_transformed)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
