import pandas as pd
import statsmodels.api as sm

# load training data from CSV file
df = pd.read_csv('training.csv')

# split data into dependent and independent variables
X = df.drop(['ONCOGENIC'], axis=1) # independent variables
y = df['ONCOGENIC'] # dependent variable
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())




