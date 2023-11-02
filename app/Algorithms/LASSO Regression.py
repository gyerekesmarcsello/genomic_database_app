import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("training.csv")
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred.round())
test_accuracy = accuracy_score(y_test, y_test_pred.round())

print("Model name: Lasso Regression")
print("Training loss:", lasso.score(X_train, y_train))
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
