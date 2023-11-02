import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_csv('training.csv')

train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)

x_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
x_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()

x_train = x_train.reshape(x_train.shape[0], 13, 2)
x_test = x_test.reshape(x_test.shape[0], 13, 2)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential(
    [
        layers.SimpleRNN(32, input_shape=(13, 2)),
        layers.Dense(6, activation='softmax')
    ]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Testing Accuracy: {test_accuracy}")