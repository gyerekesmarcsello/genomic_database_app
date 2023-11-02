import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the CSV dataset using pandas
dataset = pd.read_csv('training.csv')

# Split the dataset into training and testing data
train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)


X_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
X_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Evaluate model performance
accuracy = lr_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
