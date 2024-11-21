import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the training dataset
data = pd.read_csv('training.csv')

X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

ica = FastICA()
X_ica = ica.fit_transform(X)

clf = DecisionTreeClassifier()
clf.fit(X_ica, y)

y_pred = clf.predict(X_ica)
accuracy = accuracy_score(y, y_pred)

print('Independent Component Analysis (ICA)')
print('Training Loss: N/A')
print('Accuracy: %.2f' % accuracy)

