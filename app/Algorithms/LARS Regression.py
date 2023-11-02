import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("training.csv")
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred.round())
test_accuracy = accuracy_score(y_test, y_test_pred.round())

print("Elastic Net Regression Model")
print("Training Loss:", model.score(X_train, y_train))
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
