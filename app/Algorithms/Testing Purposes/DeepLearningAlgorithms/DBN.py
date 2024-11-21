import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('training.csv')

train_data, test_data = train_test_split(dataset, test_size=0.2)

x_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
x_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

num_visible = 26
num_hidden1 = 122
num_hidden2 = 13
num_hidden3 = 2
num_classes = 6

model = Sequential()
model.add(Dense(num_hidden1, input_shape=(num_visible,), activation='relu'))
model.add(Dense(num_hidden2, activation='relu'))
model.add(Dense(num_hidden3, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

num_epochs = 6
batch_size = 32
history = model.fit(x_train, keras.utils.to_categorical(y_train), batch_size=batch_size, epochs=num_epochs, validation_split=0.1)

train_loss, train_accuracy = model.evaluate(x_train, keras.utils.to_categorical(y_train))
print(f"Training Accuracy: {train_accuracy}")

test_loss, test_accuracy = model.evaluate(x_test, keras.utils.to_categorical(y_test))
print(f"Testing Accuracy: {test_accuracy}")