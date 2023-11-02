import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the data
data = pd.read_csv("training.csv")
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict the target variable for the training and test sets
y_train_pred = ridge_model.predict(X_train)
y_test_pred = ridge_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred.round())
test_accuracy = accuracy_score(y_test, y_test_pred.round())
train_loss = mean_squared_error(y_train, ridge_model.predict(X_train))

print("Model Name: Ridge Regression")
print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Training Loss: {:.2f}".format(train_loss))
