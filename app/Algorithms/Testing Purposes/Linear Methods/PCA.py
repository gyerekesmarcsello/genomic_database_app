import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('training.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('ONCOGENIC', axis=1), data['ONCOGENIC'], test_size=0.2, random_state=42)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_pca, y_train)
y_pred_train = clf.predict(X_train_pca)
y_pred_test = clf.predict(X_test_pca)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Model: Logistic Regression with PCA")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
