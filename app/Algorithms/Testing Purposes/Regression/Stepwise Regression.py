import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('ONCOGENIC', axis=1),
                                                    df['ONCOGENIC'],
                                                    test_size=0.2,
                                                    random_state=42)

# Stepwise regression
selected_features = []
for i in range(len(X_train.columns)):
    remaining_features = list(set(X_train.columns) - set(selected_features))
    best_score = 0
    for feature in remaining_features:
        model = sm.OLS(y_train, sm.add_constant(X_train[selected_features + [feature]])).fit()
        score = model.rsquared_adj
        if score > best_score:
            best_score = score
            best_feature = feature
    selected_features.append(best_feature)

model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()

# Predictions and accuracy
y_pred = model.predict(sm.add_constant(X_test[selected_features]))
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy:", accuracy)

print("Model:", model.summary())
print("Training loss:", model.mse_resid)
