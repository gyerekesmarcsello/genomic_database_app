import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
df = pd.read_csv('training.csv')

# Set a threshold for variance
threshold = 0.1

# Create a VarianceThreshold object
vt = VarianceThreshold(threshold)

# Fit the object to the dataset
vt.fit(df)

# Get the indices of the features that have a variance greater than the threshold
indices = vt.get_support(indices=True)

# Print the selected features
selected_features = df.columns[indices]
print(selected_features)
