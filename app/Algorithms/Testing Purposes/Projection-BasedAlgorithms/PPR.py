import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset using pandas
data = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Instantiate a PCA object to reduce the dimensionality of the data
pca = PCA(n_components=2)

# Instantiate a LinearRegression object to fit the PPR model
ppr_model = LinearRegression()

# Create a pipeline to perform PCA followed by PPR
ppr_pipeline = make_pipeline(pca, ppr_model)

# Fit the pipeline to the training data
ppr_pipeline.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = ppr_pipeline.predict(X_test)

# Compute the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('MSE: {:.2f}'.format(mse))
