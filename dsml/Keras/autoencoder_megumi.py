from tensorflow.contrib import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import adam, adadelta, rmsprop, SGD, adagrad
import numpy as np
import matplotlib.pyplot as plt


input_layer = Input(shape=(200 * 200 * 3,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dropout(0.3)(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoder = Model(input_layer, encoded)


decoded = Dense(128, activation='relu')(encoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(200 * 200 * 3, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)


decoder_input = Input(shape=(16,))
decoded = decoder_input
for layer in autoencoder.layers[-3:]:
    decoded = layer(decoded)
decoder = Model(decoder_input, decoded)

autoencoder.compile(loss='binary_crossentropy',
                    optimizer=rmsprop(lr=0.0005),
                    metrics=['accuracy'])

megumi = plt.imread('/home/dorian/Desktop/megumi.jpg') / 256.
plt.imshow(megumi)
x = megumi.reshape(1, -1)
autoencoder.fit(x, x, epochs=100000)

