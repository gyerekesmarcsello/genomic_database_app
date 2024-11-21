import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('training.csv')

# Separate target variable and features
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select top k features using chi-square test
k = 5
selector = SelectKBest(chi2, k=k)
selector.fit(X_train, y_train)

# Transform the data to include only the selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train a decision tree classifier on the selected features
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_selected, y_train)

# Predict on test set and calculate accuracy
y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
