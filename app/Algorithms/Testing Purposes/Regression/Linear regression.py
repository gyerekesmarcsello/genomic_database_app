import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("training.csv")

X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

training_loss = mean_squared_error(y_train, y_train_pred)
testing_loss = mean_squared_error(y_test, y_test_pred)
training_accuracy = r2_score(y_train, y_train_pred)
testing_accuracy = r2_score(y_test, y_test_pred)

# Print the name of the model
print("Linear Regression")

# Print the training loss and accuracy
print("Training Loss:", training_loss)
print("Training Accuracy:", training_accuracy)

# Print the testing loss and accuracy
print("Testing Loss:", testing_loss)
print("Testing Accuracy:", testing_accuracy)
