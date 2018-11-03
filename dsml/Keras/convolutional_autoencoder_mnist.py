from tensorflow.contrib import keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist

input_layer = Input(shape=(28, 28, 1))

# 28 x 28 x 1
_ = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
# 26 x 26 x 8
_ = MaxPooling2D((2, 2), padding='same')(_)
# 13 x 13 x 8
_ = Conv2D(8, (3, 3), activation='relu', padding='same')(_)
# 11 x 11 x 64
_ = MaxPooling2D((2, 2), padding='same')(_)
# 
_ = Conv2D(8, (3, 3), activation='relu', padding='same')(_)
_ = MaxPooling2D((2, 2), padding='same')(_)
encoded = _

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

_ = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
_ = UpSampling2D((2, 2))(_)
_ = Conv2D(8, (3, 3), activation='relu', padding='same')(_)
_ = UpSampling2D((2, 2))(_)
_ = Conv2D(16, (3, 3), activation='relu')(_)
_ = UpSampling2D((2, 2))(_)
_ = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(_)
decoded = _

encoder = Model(input_layer, encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
