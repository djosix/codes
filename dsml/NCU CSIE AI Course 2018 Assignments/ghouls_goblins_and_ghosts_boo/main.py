import pandas as pd
import numpy as np
from tensorflow import keras

#================================================================
# Load data

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

types = ['Ghost', 'Goblin', 'Ghoul']
colors =  ['white', 'black', 'clear', 'blue', 'green', 'blood']

#================================================================
# Process features

def process_x(rows):
    return np.array([
        [*scalars, *[float(color == c) for c in colors]]
        for *scalars, color in rows])
    
def process_y(rows):
    return np.array([
        [float(type_ == t) for t in types]
        for type_ in rows])

train_x = process_x(train_data.values[:, 1:-1])
test_x = process_x(test_data.values[:, 1:])
train_y = process_y(train_data.values[:, -1])

feat_dim = train_x.shape[1]
out_dim = len(types)

#================================================================
# Feature normalization

mean = train_x.mean(0)
std = train_x.std(0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

#================================================================
# Train

model = keras.models.Sequential([
    keras.layers.Dense(
        32,
        input_shape=(feat_dim,),
        activation='sigmoid'),
    keras.layers.Dense(
        out_dim,
        activation='softmax'),
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    x=train_x,
    y=train_y,
    batch_size=64,
    epochs=192)

pred_y = [types[y] for y in model.predict(x=test_x).argmax(-1)]
result = test_data.assign(type=pred_y)
result.to_csv('pred.csv', columns=['id', 'type'], index=False)
