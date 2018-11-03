import torch
import random, itertools
from os.path import basename

import config
from dataset.utils import load_data


def load_dataset():
    cache_name = 'cache/dataset.cache'
    try:
        return torch.load(cache_name)
    except:
        # pylint: disable=E0632
        (train, dev, test) = _load('dataset/data')
        # pylint: enable=E0632
        data = {'train': train, 'dev': dev, 'test': test}
        torch.save(data, cache_name)
        return data


#====================================================================

TOKENS \
    = SOS, EOS, UNK, PAD \
    = '[sos]', '[eos]', '[unk]', '[pad]'

#====================================================================

class TOEFL:
    def __init__(self, samples):
        self.samples = samples
    
    def batches(self, batch_size=64):
        samples = list(self.samples)
        random.shuffle(samples)
        for i in range(len(samples) // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = samples[start:end]
            yield _batch(batch)


class WordVec:
    _instance = None

    @staticmethod
    def instance():
        if not WordVec._instance:
            print('Loading word vectors')
            WordVec._instance = WordVec()
        return WordVec._instance

    @staticmethod
    def convert(words):
        return WordVec.instance().to_vectors(words)

    def __init__(self):
        path = config.wordvec_path
        n_words = config.n_words
        try:
            cache_name = 'cache/{}-{}.cache'.format(basename(path),
                                                    config.n_words)
            state = torch.load(cache_name)
            self.table = state['table']
            self.n_words = state['n_words']
            self.dim = state['dim']
        except:
            self.table = {}
            self.n_words = n_words
            n_tokens = len(TOKENS)
            with open(path) as f:
                size, dim = map(int, f.readline().split())
                assert n_words <= size
                self.dim = dim + n_tokens
                for _ in range(n_words):
                    [word, *vector] = f.readline().split()
                    assert len(vector) == dim
                    vector = list(map(float, vector))
                    vector += [0.] * n_tokens
                    self.table[word] = vector
            for i, token in enumerate(TOKENS):
                vector = [0.] * self.dim
                vector[-i - 1] = 1.
                self.table[token] = vector
            torch.save({
                'table': self.table,
                'n_words': self.n_words,
                'dim': self.dim
            }, cache_name)

    def to_vector(self, word):
        try:
            return self.table[word]
        except:
            return self.table[UNK]

    def to_vectors(self, words):
        return [self.to_vector(word) for word in words]


#====================================================================

def _load(path):
    dataset = load_data('dataset/data')
    toefl = []
    for data in dataset:
        samples = []
        for _, sample in data.items():
            # Use option index as the answer
            sample['answer'] = sample['options'].index(sample['answer'])
            sample['question'] = [SOS, *sample['question'], EOS]
            sample['sentences'] = [[SOS, *sentence, EOS]
                                   for sentence in sample['sentences']]
            sample['options'] = [[SOS, *option, EOS]
                                 for option in sample['options']]
            samples.append(sample)
        toefl.append(TOEFL(samples))
    return toefl


def _batch(samples):
    # Part A: data layout
    story = [] # merge sentences to a story
    query = []
    options = ([], [], [], [])
    answer = []
    for sample in samples:
        story.append(list(itertools.chain(*sample['sentences'])))
        query.append(sample['question'])
        for i, option in enumerate(sample['options']):
            options[i].append(option)
        answer.append(sample['answer'])
    # Part B: padding, word vectors, and converting to tensor
    b_story = _seqs_to_tensor(story) # seqs, seqlens
    b_query = _seqs_to_tensor(query) # seqs, seqlens
    b_options = ([], []) # seqs[4], seqlens[4]
    for option in options:
        option, option_len = _seqs_to_tensor(option)
        b_options[0].append(option)
        b_options[1].append(option_len)
    b_answer = torch.LongTensor(answer)
    return {
        'story': b_story, # (tensor[maxlen, batch], lens)
        'query': b_query, # (tensor[maxlen, batch], lens)
        'options': b_options, # list of (tensor[maxlen, batch], lens)
        'answer': b_answer # tensor !!!
    }


def _seqs_to_tensor(seqs):
    seqlens = list(map(len, seqs))
    # Here `seqs` become step-first
    seqs = list(itertools.zip_longest(*seqs, fillvalue=PAD))
    # Convert words to vectors
    seqs = [WordVec.convert(seq) for seq in seqs]
    seqs = torch.FloatTensor(seqs) # embeddings (float)
    seqlens = torch.LongTensor(seqlens)
    return seqs, seqlens

