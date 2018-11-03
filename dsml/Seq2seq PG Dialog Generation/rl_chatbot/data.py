"""
Dataset & preprocessing.
"""

import os, random, itertools, re, jieba
from collections import Counter

import numpy as np
import torch
from torch.autograd import Variable

from . import config
from . import utils


# Tokens
NIL_TOKEN, NIL_INDEX = '@', 0   # for words not in dictionary
BOS_TOKEN, BOS_INDEX = '(', 1   # begining of a sentence
EOS_TOKEN, EOS_INDEX = ')', 2   # end of a sentence
PAD_TOKEN, PAD_INDEX = ';', 3   # padding
TOKENS = [NIL_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]



# class Lang:
#     def __init__(self, tokenizer=None):
#         self._tokens = []
#         self._tokenizer = tokenizer
    
#     def add_words(self, words):
#         self._tokens += words
    
#     def build_dictionary(self, dict_size):
#         words = utils.progress('Building Dictionary', self._tokens)
#         words = list(map(lambda t: t[0], Counter(words).most_common()))
#         words = [*TOKENS, *words]
#         self.n_words = min(dict_size, len(words))
#         words = words[:self.n_words]
#         self._words = {i: word for i, word in enumerate(words)}
#         self._indexes = {word: i for i, word in enumerate(words)}
#         self._tokens = None # dereference

#     def tokenize(self, sentence):
#         assert self._tokenizer is not None
#         return self._tokenizer(sentence)

#     def index(self, word):
#         return self._indexes.get(word, NIL_INDEX)
    
#     def word(self, index):
#         return self._words.get(index, NIL_TOKEN)

#     def indexes(self, words):
#         return [self.index(word) for word in words
#                 if config.USE_NIL or word in self._indexes.keys()]
    
#     def words(self, indexes):
#         return [self.word(index) for index in indexes
#                 if config.USE_NIL or index in self._words.keys()]
    

class Corpus:

    def __init__(self, sentences, dict_size, tokenizer=list, target='next'):
        """
        Arguments:
        - `sentences`: List of sentences (str).
        - `dict_size` (int): Dictionary size.
        - `tokenizer` (function): Tokenizer for sentences.
        - `target` (str): 'next' or 'self'.
        """
        self.tokenize = tokenizer # a method to call
        self._tokenize_sentences(sentences)
        self._build_dictionary(dict_size)
        self._build_training_pairs(target)
        self.sentences = None # remove reference (not saved)
    
    def to_indexes(self, words):
        return [self.indexes.get(word, NIL_INDEX) for word in words
                if config.USE_NIL or self.indexes.get(word) is not None]

    def to_words(self, indexes):
        return [self.words.get(index, NIL_TOKEN) for index in indexes
                if config.USE_NIL or self.words.get(index) is not None]

    def sentences_to_index_seqs(self, sentences):
        return list(map(self.to_indexes, map(self.tokenize, sentences)))

    def index_seqs_to_words(self, seqs):
        return [self.to_words(seq) for seq in seqs]

    def index_seqs_to_sentences(self, seqs, sep=' ', token=False):
        return [sep.join(filter(lambda w: token or w not in TOKENS,
                self.to_words(seq))) for seq in seqs]

    def index_seqs_variable(self, seqs, bos=False, eos=False):
        # seqs (sorted): [batch, step], padded: [step, batch]
        prefix = [BOS_INDEX] if bos else []
        postfix = [EOS_INDEX] if eos else []
        seqs = [[*prefix, *seq, *postfix] for seq in seqs]
        seq_lens = [len(seq) for seq in seqs]
        padded = list(itertools.zip_longest(*seqs, fillvalue=PAD_INDEX))
        return Variable(torch.LongTensor(padded)), seq_lens


    def batches(self, size, method='shuffle'):
        """
        Arguments:
        - `method` (str): 'shuffle' or 'replace'
        """
        pairs = self.pairs.copy()
        batches = []

        if method in ['shuffle', 'replace']:
            batch_indexes = list(range(0, len(pairs), size))
            if method == 'replace':
                batch_indexes = np.random.choice(
                    batch_indexes, len(batch_indexes), replace=True)
                print(len(batch_indexes))
            else:
                np.random.shuffle(pairs)
            for i in utils.progress('Generating batches', batch_indexes):
                end = i + size
                if end > len(pairs): break
                batch = pairs[i:end]
                batch.sort(key=lambda p: len(p[0]), reverse=True)
                xs, ys = zip(*batch)
                batches.append([
                    self.index_seqs_variable(xs),
                    self.index_seqs_variable(ys, eos=True)
                ])

        else:
            assert False, 'Unknown method: {}'.format(method)

        return batches


    def _tokenize_sentences(self, sentences):
        # Tokenize sentences
        sentences = utils.progress('Preprocessing', sentences)
        sentences = map(self.tokenize, sentences)
        self.sentences = list(sentences)


    def _build_dictionary(self, dict_size):
        # Build dictionary of `dict_size`
        tokens = [token for sentence in self.sentences for token in sentence]
        tokens = utils.progress('Building dictionary', tokens)
        words = list(map(lambda t: t[0], Counter(tokens).most_common()))
        words = [*TOKENS, *words] # add speical tokens...
        self.n_words = min(dict_size, len(words))
        words = words[:self.n_words]
        self.words = {i: word for i, word in enumerate(words)}
        self.indexes = {word: i for i, word in enumerate(words)}


    def _build_training_pairs(self, target):
        if target in ['next', 'prev']:
            # Predict next sentence
            if target == 'next':
                pairs = list(zip(self.sentences[:-1], self.sentences[1:]))
            else:
                pairs = list(zip(self.sentences[1:], self.sentences[:-1]))
            pairs = utils.progress('Building training pairs', pairs)
            self.pairs = []
            for x, y in pairs:
                x, y = self.to_indexes(x), self.to_indexes(y)
                if not config.TRAIN_NIL and (NIL_INDEX in x or NIL_INDEX in y):
                    continue
                if len(x) < config.MIN_INPUT_LEN or len(x) > config.MAX_INPUT_LEN:
                    continue
                if len(y) < config.MIN_TARGET_LEN or len(y) > config.MAX_TARGET_LEN:
                    continue
                self.pairs.append([x, y])

        elif target == 'self':
            # For autoencoder
            sentences = utils.progress('Building training pairs', self.sentences)
            self.pairs = []
            for x in sentences:
                x = self.to_indexes(x)
                if not config.TRAIN_NIL and (NIL_INDEX in x):
                    continue
                if len(x) < config.MIN_INPUT_LEN or len(x) > config.MAX_INPUT_LEN:
                    continue
                self.pairs.append([x, x])

        else:
            assert False, 'Unknown target type: {}'.format(target)
    
