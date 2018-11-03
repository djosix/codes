import torch
from torch.autograd import Variable

import os, tqdm, jieba, re, itertools
tqdm.tqdm.monitor_interval = 0

from . import config
from . import data



#==================================================
# Useful

def progress(desc, it):
    return tqdm.tqdm(it, desc=desc, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}')

def load(directory, name):
    path = os.path.join(directory, name)
    return torhc.load(path)

#==================================================
# Corpus loader

def load_corpus(name, dict_size, tokenizer, target, file_filter=lambda f: True, use_cache=True):
    cache_path = os.path.join(config.CACHE_DIR, '{}_{}_{}_{}_{}_{}_{}.cache'.format(
        name, dict_size, target, config.MIN_INPUT_LEN, config.MAX_INPUT_LEN,
        config.MIN_TARGET_LEN, config.MAX_TARGET_LEN))
    if use_cache and os.path.isfile(cache_path):
        try:
            print('Loading dataset from ' + cache_path)
            corpus = torch.load(cache_path)
            print('Dataset is loaded from cache')
            return corpus
        except FileNotFoundError:
            print('Failed')

    corpus_path = os.path.join(config.DATA_DIR, name)
    assert os.path.isdir(corpus_path), 'Path should be a directory'
    print('Loading dataset from ' + corpus_path)
    paths = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(corpus_path, followlinks=True)
        for filename in filter(file_filter, filenames)
    ]
    sentences = [line.strip() for path in paths for line in open(path).readlines()]
    corpus = data.Corpus(sentences, dict_size, tokenizer=tokenizer, target=target)
    print('Done, saving to ' + cache_path)
    torch.save(corpus, cache_path)
    print('Saved')
    return corpus


#==================================================
# Sentence tokenizers

def chinese_tokenizer(line):
    line = re.sub(r'，', ' ', line)
    line = re.sub(r'[^\u4e00-\u9fff ]', '', line)
    sentence = list('，'.join(line.split()))
    return sentence


def jieba_tokenizer(line):
    line = re.sub(r'，', ' ', line)
    line = re.sub(r'[^\u4e00-\u9fff ]', '', line)
    line = '，'.join(line.split())
    sentence = list(jieba.cut(line))
    return sentence


def english_tokenizer(line):
    line = line.lower()
    line = re.sub(r'[^a-z0-9 ]', '', line)
    return line.split()


#==================================================
# Batches generators

def general_batches_generator(corpus, size, method):
    while True:
        yield corpus.batches(size, method)


def overfit_batches_generator(corpus, size, method, window=100, speed=5):
    while True:
        # Do a shuffle
        print('Renew batches')
        batches = corpus.batches(64, method)
        end = len(batches) - window
        for i in range(0, end, speed):
            # Pick some batches
            j = i + window
            print('Batches: {} to {} end {}'.format(i, j, end))
            yield batches[i:j]
