import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load CSV file
df = pd.read_csv('training.csv')

# Split data into features and target
X = df.drop('ONCOGENIC', axis=1)
y = df['ONCOGENIC']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate information gain for each feature
ig_scores = mutual_info_classif(X_train, y_train)

# Select top features based on information gain
num_top_features = 10
top_features = X.columns[ig_scores.argsort()[::-1][:num_top_features]]

# Train a classifier using selected features
clf = GaussianNB()
clf.fit(X_train[top_features], y_train)

# Make predictions on test set and calculate accuracy
y_pred = clf.predict(X_test[top_features])
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
