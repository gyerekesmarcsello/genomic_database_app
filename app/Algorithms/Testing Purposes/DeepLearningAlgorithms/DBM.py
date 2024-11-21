import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
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
num_hidden = 122
num_iterations = 500

input_layer = Input(shape=(num_visible,))
hidden_layer = Dense(num_hidden, activation='sigmoid')(input_layer)
output_layer = Dense(num_visible, activation='sigmoid')(hidden_layer)

rbm = Model(inputs=input_layer, outputs=output_layer)
rbm.compile(optimizer=Adam(), loss='mean_squared_error')

for i in range(num_iterations):
    rbm.fit(x_train, x_train, batch_size=32, epochs=1, verbose=0)
    print('Iteration:', i)

dbn_layers = [hidden_layer]
for i in range(1, len(rbm.layers)):
    dbn_layers.append(rbm.layers[i])

dbn = Model(inputs=input_layer, outputs=dbn_layers)
dbn.compile(optimizer=Adam(), loss='mean_squared_error')

dbn.pop()
for layer in dbn.layers:
    layer.trainable = False

classifier = Sequential()
classifier.add(dbn)
classifier.add(Flatten())
classifier.add(Dense(10, activation='softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
classifier.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

score = classifier.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])