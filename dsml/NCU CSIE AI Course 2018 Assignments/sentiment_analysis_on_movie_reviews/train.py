import json, base64, bites, time, pickle
from tensorflow import keras
from data import load_dataset, load_embedding
from model import ModelBuilder
import numpy as np

# pylint: disable=E1101

train_x, train_y, test_x = load_dataset()
embed_init = load_embedding()

def train(**config):
    assert config['arch'] in ('CNN', 'RNN')
    builder = ModelBuilder(**config)
    builder.model_options['metrics'] = ['accuracy']
    if config['arch'] == 'CNN':
        model = builder.build_cnn_model()
    else:
        model = builder.build_rnn_model()
    # model.summary()
    print(builder.get_config())
    history = model.fit(x=[train_x],
                        y=[train_y],
                        batch_size=config.get('batch_size', 256),
                        epochs=config.get('epochs', 5))

    thash = bites.Bs.from_int(int(time.time() * 100))[:3].rev().hex()
    valaccs = np.array(history.history.get('val_acc', [0]))
    maxi = valaccs.argmax()
    valacc = int(valaccs[maxi] * 100)
    tag = '-' + config['tag'] if 'tag' in config else ''
    name = 'p{}-{}-{:02d}@{}{}'.format(config['arch'], thash, valacc, maxi, tag)
    model.save('{}.save'.format(name))

train(arch='CNN', embed_init=embed_init, embed_train=True, tag='wv1', epochs=3)
train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(32, 0.3)], tag='wv1:l32d0.3', epochs=2)
