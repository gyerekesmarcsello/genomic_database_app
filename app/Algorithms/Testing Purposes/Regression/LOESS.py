import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("training.csv")
X = data.drop(columns=["ONCOGENIC"])
y = data["ONCOGENIC"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lowess = sm.nonparametric.lowess
z = lowess(y_train, X_train.values[:, 0], frac=0.3)

y_train_pred = z[:, 1] >= 0.5
train_accuracy = sum(y_train == y_train_pred) / len(y_train)
y_test_pred = lowess(y_test, X_test.values[:, 0], frac=0.3)[:, 1] >= 0.5
test_accuracy = sum(y_test == y_test_pred) / len(y_test)

print("Model: LOESS")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
