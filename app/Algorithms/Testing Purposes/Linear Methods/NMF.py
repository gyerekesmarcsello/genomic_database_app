import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

# Load the training data from a CSV file
data = pd.read_csv("training.csv")

X = data.drop("ONCOGENIC", axis=1).apply(lambda x: np.maximum(0, x))
y = data["ONCOGENIC"]
model = NMF(n_components=10, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
training_loss = mean_squared_error(X, W.dot(H))
y_pred = W.dot(H)[:, 0]
accuracy = 1 - mean_squared_error(y, y_pred) / mean_squared_error(y, y.mean())

# Print the model name, training loss, and accuracy
print("Model: NMF")
print("Training Loss:", training_loss)
print("Accuracy:", accuracy)
