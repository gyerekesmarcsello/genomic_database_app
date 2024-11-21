from sklearn import *
import pandas as pd
import numpy as np
from pyearth import Earth
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------- Ensemble Algorithms ---------------

# ------ Bootstrapped Aggregation ------

# Create a decision tree regressor as the base estimator
base_estimator = DecisionTreeRegressor()

# Create the bagging regressor
bagging = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Fit the model to the data
bagging.fit(X, y)

# ------ Weighted Average ------

def weighted_average(data, weights):
    """Computes the weighted average of a list of data points.
    
    Args:
    data (list): List of data points to be averaged.
    weights (list): List of weights corresponding to each data point.
    
    Returns:
    float: Weighted average of the data points.
    """
    # Ensure that the data and weights lists have the same length
    if len(data) != len(weights):
        raise ValueError("Data and weights lists must have the same length.")
    
    # Compute the weighted sum and sum of the weights
    weighted_sum = 0
    sum_of_weights = 0
    for i in range(len(data)):
        weighted_sum += data[i] * weights[i]
        sum_of_weights += weights[i]
    
    # Compute the weighted average
    weighted_average = weighted_sum / sum_of_weights
    
    return weighted_average

# ------ Gradient Boosting ------

gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)

# Train the model on the training data
gb_reg.fit(X_train, y_train)

# Predict on the test data
y_pred = gb_reg.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

# ------ AdaBoost ------

# Create an AdaBoost classifier with decision tree as base estimator
ada = AdaBoostClassifier(n_estimators=50, base_estimator=DecisionTreeClassifier(max_depth=1))

# Train the model
ada.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ada.predict(X_test)


# ----- Gradient Boosting Machines (GBM) -----

# Create DMatrix objects for training and testing sets
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Specify the parameters for the model
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the test set
y_pred = model.predict(dtest)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)


# ----- LightGBM -----

# Convert data into LightGBM dataset format
train_data = lgb.Dataset(x_train, label=y_train)

# Set parameters for LightGBM model
params = {'objective': 'regression', 'metric': 'mse'}

# Train LightGBM model
num_round = 100
model = lgb.train(params, train_data, num_round)

# Predict using LightGBM model
y_pred = model.predict(x_test)

print(y_pred)

# ----- XGBoost -----

# Create the DMatrix objects for XGBoost
dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

# Set the hyperparameters
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'colsample_bytree': 0.5,
    'alpha': 0.1
}

# Train the model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
predictions = model.predict(dtest)

# Calculate the mean squared error
mse = mean_squared_error(test_y, predictions)
print("Mean Squared Error: ", mse)

# ----- Catboost -----

# Create CatBoostRegressor model
model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))

# ----- Gradient Boosted Regression Trees

# Define the model
model = GradientBoostingRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# ----- Stacked Generalization -----

# Create the dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
svm = SVR()

# Define the meta model
meta_model = LinearRegression()

# Create the stacking regressor
stacking_regressor = StackingCVRegressor(regressors=(lr, dt, rf, svm), meta_regressor=meta_model, cv=KFold(n_splits=5, shuffle=True, random_state=42), shuffle=False)

# Train the stacking regressor
stacking_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = stacking_regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: %.2f" % mse)

# ----- Random Forest -----

# Create a Random Forest model with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
rf.fit(X_train, y_train)

# Predict the test data
y_pred = rf.predict(X_test)

# Calculate the R-squared score of the model
r_squared = rf.score(X_test, y_test)
print("R-squared score:", r_squared)

# ------------- Regression Models -----------------

# Linear Regression Model
model = LinearRegression()

# ------ Ordinary Least Squares --------
model.fit(x, y)

x_test = np.array([[3, 5], [4, 6]])
y_pred = model.predict(x_test)

print(y_pred)


# ------ Logistic Regression ------

# Creating a logistic regression model and fit it to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")


# ------ Stepwise Regression --------

# Use RFE with stepwise selection to select the best features
selector = RFE(model, step=1)
selector = selector.fit(x, y)

# Print the selected features and their ranking
print("Selected Features:")
for i in range(len(selector.support_)):
    if selector.support_[i]:
        print(f"Feature {i+1}, rank {selector.ranking_[i]}")


# ------ Multivariate Adaptive Regression Splines (MARS) --------

# Fit a MARS model to the training data
# The PyEarth library provides an implementation of MARS that is optimized for performance and has additional features such as automatic feature selection and interaction detection.
model = Earth(max_degree=2, max_terms=10, penalty=3.0, endspan=5)
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"R^2 score: {score}")


# ------ Locally Estimated Scatterplot Smoothing (LOESS) --------

# Fit LOESS model
lowess = sm.nonparametric.lowess(y, x, frac=0.3)

# extract the smoothed x and y values
y_smoothed = lowess[:, 1]
x_smoothed = lowess[:, 0]

# ----- Classification and Regression Tree -----

# Create and train the decision tree regressor model
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

# ----- Iterative Dichotomiser 3 (ID3) -----

class Node:
    def __init__(self, is_leaf=False, label=None, feature=None, threshold=None):
        self.is_leaf = is_leaf
        self.label = label
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        
class ID3:
    def __init__(self, max_depth=None, min_samples_leaf=1, regression=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.regression = regression
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def predict(self, X):
        return np.array([self._predict_tree(self.root, x) for x in X])
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # stopping criterion
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_leaf:
            if self.regression:
                return Node(is_leaf=True, label=np.mean(y))
            else:
                unique, counts = np.unique(y, return_counts=True)
                label = unique[np.argmax(counts)]
                return Node(is_leaf=True, label=label)
        
        if self.regression:
            impurity_func = self._variance_reduction
        else:
            impurity_func = self._entropy
            
        best_impurity = float("inf") if self.regression else -float("inf")
        best_feature = None
        best_threshold = None
        
        # loop through all features and thresholds to find the best split
        for i in range(n_features):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                impurity = impurity_func(y, X[:, i], threshold)
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = i
                    best_threshold = threshold
        
        # stopping criterion if no split improves the impurity
        if best_feature is None or best_impurity == float("inf") or best_impurity == -float("inf"):
            if self.regression:
                return Node(is_leaf=True, label=np.mean(y))
            else:
                unique, counts = np.unique(y, return_counts=True)
                label = unique[np.argmax(counts)]
                return Node(is_leaf=True, label=label)
        
        # split the node and grow subtrees
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(is_leaf=False, feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _predict_tree(self, node, x):
        if node.is_leaf:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_tree(node.left, x)
        else:
            return self._predict_tree(node.right, x)
        
        def entropy(y):
            left_idxs = x <= threshold
        right_idxs = x > threshold
        left_p = np.mean(y[left_idxs])
        right_p = np.mean(y[right_idxs])
        p = np.array([left_p, right_p])
        p = p[p > counts] = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

def information_gain(X, y, feature_idx):
    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    p = counts / np.sum(counts)
    gain = entropy(y)
    for i in range(len(values)):
        subset = y[X[:, feature_idx] == values[i]]
        gain -= p[i] * entropy(subset)
    return gain

def id3(X, y, features):
    # Base case 1: If all labels are the same, return that label
    if len(np.unique(y)) == 1:
        return y[0]
    # Base case 2: If there are no features left to split on, return the most common label
    if len(features) == 0:
        return np.bincount(y).argmax()
    # Find the feature with the highest information gain
    gains = [information_gain(X, y, feature_idx) for feature_idx in features]
    best_feature_idx = features[np.argmax(gains)]
    tree = {best_feature_idx: {}}
    # Remove the best feature from the feature list
    remaining_features = [f for f in features if f != best_feature_idx]
    # Recursively build the tree
    for value in np.unique(X[:, best_feature_idx]):
        value = int(value)
        subset = X[X[:, best_feature_idx] == value]
        sub_y = y[X[:, best_feature_idx] == value]
        subtree = id3(subset, sub_y, remaining_features)
        tree[best_feature_idx][value] = subtree
    return tree

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
tree = id3(X, y, [0, 1])
print(tree)