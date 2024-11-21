import pandas as pd
from pyearth import Earth

# Load the dataset
data = pd.read_csv('training.csv')

# Split the dataset into features and target
X = data.drop(columns=['ONCOGENIC'])
y = data['ONCOGENIC']

model = Earth(max_terms=10, max_degree=3, penalty=3.0)
model.fit(X, y)

accuracy = model.score(X, y)
print("MARS model using py-earth")
print("Training loss: ", model.mse_)
print("Accuracy: ", accuracy)
