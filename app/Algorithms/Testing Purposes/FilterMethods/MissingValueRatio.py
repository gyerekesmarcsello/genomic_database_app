import pandas as pd

# Load the dataset
df = pd.read_csv('training.csv')

# Calculate the percentage of missing values in each feature
missing_values = df.isna().mean()

# Sort the features by the percentage of missing values
missing_values = missing_values.sort_values(ascending=False)

# Print the percentage of missing values for each feature
print(missing_values)
