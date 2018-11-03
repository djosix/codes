from tensorflow.contrib import keras

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

input_layer = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

encoder = Model(input_layer, encoded)
encoder.summary()

autoencoder = Model(input_layer, decoded)
autoencoder.summary()

autoencoder.compile(loss='binary_crossentropy',
                    optimizer='adam',
#                    shuffle=True,
                    metrics=['accuracy'])

(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

history = autoencoder.fit(x_train, x_train, epochs=4, batch_size=512)

y = history.history['loss']
x = np.arange(len(y))
plt.plot(x, y)

plt.show()
