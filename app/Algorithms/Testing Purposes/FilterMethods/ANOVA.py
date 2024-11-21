import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the CSV dataset
data = pd.read_csv('training.csv')

# split the data into features and target variable
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# perform feature selection using ANOVA
k = 10  # number of top features to select
anova = SelectKBest(score_func=f_classif, k=k)
X_anova = anova.fit_transform(X, y)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_anova, y, test_size=0.2, random_state=42)

# train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
