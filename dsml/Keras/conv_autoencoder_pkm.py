import numpy as np
import matplotlib.pyplot as plt
import os

path = 'pkm/'


imgs = [plt.imread(path + p) / 256.
    for p in os.listdir(path)
        if p.endswith('.jpg')]
imgs = np.array(imgs).reshape(-1, 64, 64, 3)
    
for i, img in enumerate(imgs):
    plt.subplot(4, 6, i + 1)
    plt.imshow(img)
    plt.show()


from tensorflow.contrib import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import adadelta, adam, rmsprop, SGD, adagrad
from keras import backend as K


inputs = Input(shape=(64, 64, 3))
x = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoder = Model(inputs, encoded)
encoder.summary()

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs, decoded)
autoencoder.summary()
autoencoder.compile(optimizer=adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

decoder_inputs = Input(shape=[1, 1, 32])
x = decoder_inputs
for layer in autoencoder.layers[-13:]:
    x = layer(x)
decoder = Model(decoder_inputs, x)
decoder.summary()



history = autoencoder.fit(imgs, imgs,
                          epochs=500000,
                          batch_size=128)

y = history.history['loss']
x = np.arange(len(y))
plt.plot(x, y)
plt.show()
