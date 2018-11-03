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


import tensorflow as tf
from tensorflow.contrib import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import adadelta, adam, rmsprop, SGD, adagrad
from keras import backend as K


inputs = Input(shape=(64, 64, 3))
x = Conv2D(128, (5, 5), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(23, (3, 3), activation='relu', padding='same')(x)
outputs = MaxPooling2D((2, 2), padding='same')(x)

convnet = Model(inputs, outputs)
convnet.summary()

convnet.compile(optimizer=adam(),
                loss='binary_crossentropy',
                metrics=['accuracy'])

labels = np.eye(len(imgs))

convnet.fit(imgs, labels.reshape(-1, 1, 1, 23),
            epochs=80)