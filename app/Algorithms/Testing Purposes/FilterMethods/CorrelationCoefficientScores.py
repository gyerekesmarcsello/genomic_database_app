import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Load the dataset
df = pd.read_csv('training.csv')

# Separate the target variable from the features
X = df.drop('ONCOGENIC', axis=1)
y = df['ONCOGENIC']

# Use SelectKBest to select the top k features based on the correlation coefficient score
k = 10  # Set the number of features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)

# Print the selected features and their correlation scores
selected_features = X.columns[selector.get_support(indices=True)]
correlation_scores = selector.scores_[selector.get_support(indices=True)]
for feature, score in zip(selected_features, correlation_scores):
    print(f'{feature}: {score}')
