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
                        epochs=config.get('epochs', 5),
                        validation_split=config.get('val_split', 0.25))

    thash = bites.Bs.from_int(int(time.time() * 100))[:3].rev().hex()
    valaccs = np.array(history.history.get('val_acc', [0]))
    maxi = valaccs.argmax()
    valacc = int(valaccs[maxi] * 100)
    tag = '-' + config['tag'] if 'tag' in config else ''
    name = 'a{}-{}-{:02d}@{}{}'.format(config['arch'], thash, valacc, maxi, tag)
    with open('result/{}.json'.format(name), 'w') as f:
        json.dump(history.history, f)


train(arch='CNN')
train(arch='CNN', embed_init=embed_init, embed_train=True, tag='wv1')
train(arch='CNN', embed_init=embed_init, embed_train=True, cnn_dropout=0.7, tag='wv1:d0.7')
train(arch='CNN', embed_init=embed_init, embed_train=True, cnn_dropout=0.9, tag='wv1:d0.9')
train(arch='CNN', embed_init=embed_init, embed_train=False, tag='wv0')
train(arch='CNN', embed_init=embed_init, embed_train=True, cnn_filters=[(2, 32), (3, 32)], cnn_merge_dim=64, tag='wv0:2f32m64')
train(arch='CNN', embed_init=embed_init, embed_train=True, cnn_filters=[(2, 64), (3, 64)], cnn_merge_dim=128, tag='wv0:2f64m128')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 128), (3, 128)], cnn_merge_dim=256, tag='wv0:2f128m256')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 32), (3, 32), (4, 32)], cnn_merge_dim=64, tag='wv0:3f32m64')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 64), (3, 64), (4, 64)], cnn_merge_dim=128, tag='wv0:3f64m128')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 128), (3, 128), (4, 128)], cnn_merge_dim=256, tag='wv0:3f128m256')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 32), (3, 32), (4, 32), (5, 32)], cnn_merge_dim=64, tag='wv0:4f32m64')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_filters=[(2, 64), (3, 64), (4, 64), (5, 64)], cnn_merge_dim=128, tag='wv0:4f64m128')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_dropout=0.3, tag='wv0:d0.3')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_dropout=0.6, tag='wv0:d0.6')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_dropout=0.8, tag='wv0:d0.8')
train(arch='CNN', embed_init=embed_init, embed_train=False, cnn_dropout=0.9, tag='wv0:d0.9')
# train(arch='RNN')
# train(arch='RNN', embed_init=embed_init, embed_train=True, tag='wv1')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(64, 0.5)], tag='wv1:l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(64, 0.8)], tag='wv1:l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(32, 0.0)], tag='wv1:l32d0')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(32, 0.3)], tag='wv1:l32d0.3')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(32, 0.5)], tag='wv1:l32d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(128, 0.5)], tag='wv1:l128d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(128, 0.7)], tag='wv1:l128d0.7')
# train(arch='RNN', embed_init=embed_init, embed_train=True, rnn_layers=[(128, 0.9)], tag='wv1:l128d0.9')
# train(arch='RNN', embed_init=embed_init, embed_train=False, tag='wv0')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(64, 0.5)], tag='wv0:l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(64, 0.8)], tag='wv0:l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.0)], tag='wv0:l32d0')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.3)], tag='wv0:l32d0.3')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.5)], tag='wv0:l32d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.5)], tag='wv0:l128d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.7)], tag='wv0:l128d0.7')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.9)], tag='wv0:l128d0.9')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(64, 0.5), (64, 0.5)], tag='wv0:2l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(64, 0.8), (64, 0.8)], tag='wv0:2l64d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.0), (32, 0.0)], tag='wv0:2l32d0')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.3), (32, 0.3)], tag='wv0:2l32d0.3')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(32, 0.5), (32, 0.5)], tag='wv0:2l32d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.5), (128, 0.5)], tag='wv0:2l128d0.5')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.7), (128, 0.7)], tag='wv0:2l128d0.7')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(128, 0.9), (128, 0.9)], tag='wv0:2l128d0.9')
# train(arch='RNN', embed_init=embed_init, embed_train=False, rnn_layers=[(256, 0.9), (256, 0.9)], tag='wv0:2l256d0.9')

