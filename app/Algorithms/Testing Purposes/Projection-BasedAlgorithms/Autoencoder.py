import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset using pandas
data = pd.read_csv('training.csv')

# Split the dataset into training and testing sets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# Normalize the data to have zero mean and unit variance
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Define the shape of the input data
input_shape = X_train.shape[1]

# Instantiate an autoencoder model
autoencoder = Sequential([
    Input(shape=input_shape),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_shape)
])

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate the autoencoder model on the test data
loss = autoencoder.evaluate(X_test, X_test)
print('Loss: {:.2f}'.format(loss))

