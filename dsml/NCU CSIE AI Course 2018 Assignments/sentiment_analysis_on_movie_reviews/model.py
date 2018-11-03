import pandas as pd
import numpy as np
import pickle, os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense, Conv1D, Input, Embedding, Dropout,
    GlobalMaxPooling1D, Concatenate, LSTM,
    LeakyReLU, GRU, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from config import defaults

# pylint: disable=E1101

__all__ = ['ModelBuilder']

class ModelBuilder:
    
    def __init__(self, **kwargs):
        config = defaults.copy()
        config.update(kwargs)
        for key, value in config.items():
            setattr(self, key, value)
    
    def _input(self):
        return Input([self.input_len])

    def _embedding_layer(self, x):
        kwargs = {'input_dim': self.input_dim,
                  'output_dim': self.embed_dim,
                  'input_length': self.input_len,
                  'trainable': self.embed_train}
        if self.embed_init is not None:
            kwargs['weights'] = [self.embed_init]
        return Embedding(**kwargs)(x)
    
    def _output_layer(self, x):
        return Dense(self.output_dim, activation='softmax')(x)
    
    def _cnn_layers(self, x):
        xs = [
            Conv1D(filters=f, kernel_size=k, padding='valid', strides=1)(x)
            for k, f in self.cnn_filters]
        xs = map(LeakyReLU(0.01), xs)
        xs = map(GlobalAveragePooling1D(), xs)
        xs = list(xs)
        x = Concatenate(axis=1)(xs)
        x = Dense(self.cnn_merge_dim, activation=self.cnn_merge_act)(x)
        x = Dropout(self.cnn_dropout)(x)
        return x

    def _rnn_layers(self, x):
        N = len(self.rnn_layers)
        for i, (n, d) in enumerate(self.rnn_layers):
            x = GRU(n, return_sequences=(i < N-1))(x)
            x = Dropout(d)(x)
        return x

    def _compiled_model(self, i, o):
        m = Model(inputs=[i], outputs=[o])
        m.compile(**self.model_options)
        return m
    
    def build_cnn_model(self):
        i = self._input()
        x = self._embedding_layer(i)
        x = self._cnn_layers(x)
        o = self._output_layer(x)
        return self._compiled_model(i, o)
    
    def build_rnn_model(self):
        i = self._input()
        x = self._embedding_layer(i)
        x = self._rnn_layers(x)
        o = self._output_layer(x)
        return self._compiled_model(i, o)

    def get_config(self):
        return {key: getattr(self, key) for key in defaults}
        
