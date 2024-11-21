import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('training.csv')

# Set a threshold for correlation
threshold = 0.9

# Create a correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than threshold
correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]

# Print the selected features
selected_features = df.drop(columns=correlated_features)
print(selected_features.head())
