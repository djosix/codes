import os
from tqdm import tqdm
import numpy as np
import re

not_chinese = '[^\u4e00-\u9fff]'
stop_words = {
    '有', '所', '啊', '都', '來', '啦',
    '也', '好', '到', '的', '樣', '會',
    '在', '什', '就', '那', '\u3000',
    '個', '是', '以', '要', '怎', '麼',
    '子', '可', '人', '嗎', '這', '了'
}

""" max len
3 sentences: 270
1 sentence: 
"""

class Dataset:

    def load(fname):
        import pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self,
                 root='./dataset/training_data/subtitle_no_TC',
                 use_stop_words=False):
        self.use_stop_words = use_stop_words
        self.flist = []
        for path, dirs, files in os.walk(root):
            for f in filter(lambda s: s.endswith('.txt'), files):
                self.flist += [path + '/' + f]
        self.data = []
        self.max_seq_len = 0
        self.top_seq_lens = []
        chars = set()
        for fname in tqdm(self.flist, desc='Parsing words'):
            with open(fname, 'rt') as f:
                l1 = lambda s: (s != '')
                l2 = lambda s: self.filt(s)
                lines = list(filter(l1, map(l2, f.readlines())))
                m = max(len(line) for line in lines)
                if m > self.max_seq_len:
                    self.max_seq_len = m
                    self.top_seq_lens += [m]
                chars |= set(''.join(lines))
                self.data += lines
        self.data = np.array(self.data)
        self.charset = chars
        self.chars = list(chars)
        self.vsize = len(self.chars)
        self.dict = dict((c, i) for i, c in enumerate(self.chars))
    

    # def dialog_and_next(self, dialog_len, padding=False):
    #     n_dialog = len(self.data)
    #     while True:
    #         ss = self.data[np.random.randint(n_dialog)]
    #         ss = [self.encode_sentence(s) for s in ss]
    #         for i in range(len(ss) - dialog_len):
    #             dialog = ss[i:i+dialog_len]
    #             dialog = [c for s in dialog for c in s]
    #             s_next = ss[i+dialog_len]
    #             if padding:
    #                 dialog, l1 = self.pad_sequence(dialog, padding, return_len=True)
    #                 s_next, l2 = self.pad_sequence(s_next, padding, return_len=True)
    #                 yield (dialog, s_next), [l1, l2]
    #             else:
    #                 yield dialog, s_next
    
    # def batch_dialog_and_next(self, batch_size, dialog_len):
    #     g = self.dialog_and_next(dialog_len, padding=False)
    #     D, S, D_lens, S_lens, max_len = [], [], [], [], 0
    #     for i, (dialog, s_next) in enumerate(g):
    #         D += [dialog]
    #         S += [s_next]
    #         D_lens += [len(dialog)]
    #         S_lens += [len(s_next)]
    #         k = max(len(dialog), len(s_next))
    #         max_len = k if k > max_len else max_len
    #         if i % batch_size == batch_size - 1:
    #             D = [self.pad_sequence(s, max_len) for s in D]
    #             S = [self.pad_sequence(s, max_len) for s in S]
    #             yield (D, D_lens), (S, S_lens), max_len
    #             D, S, D_lens, S_lens, max_len = [], [], [], [], 0

    def random_batch_generator(self, batch_size, dialog_len):
        data_size = self.data.shape[0]
        while True:
            indexes = np.random.randint(data_size - dialog_len - 1, size=batch_size)
            dialogs, s_nexts, dialog_lens, s_next_lens, max_len = [], [], [], [], 0
            for i in indexes:
                *dialog, s_next = self.data[np.arange(i, i+dialog_len+1)]
                dialog = self.encode_many(dialog)
                s_next = self.encode(s_next)
                k = max(len(dialog), len(s_next))
                max_len = k if k > max_len else max_len
                dialogs.append(dialog)
                s_nexts.append(s_next)
                dialog_lens.append(len(dialog))
                s_next_lens.append(len(s_next))
            dialogs = [self.pad_sequence(dialog, max_len) for dialog in dialogs]
            s_nexts = [self.pad_sequence(s_next, max_len) for s_next in s_nexts]
            yield (dialogs, dialog_lens), (s_nexts, s_next_lens)

    def random_batch_generator_neg(self, batch_size, dialog_len):
        data_size = self.data.shape[0]
        while True:
            indexes = np.random.randint(data_size - dialog_len, size=batch_size)
            neg_idx = np.random.randint(data_size, size=batch_size)
            dialogs, s_nexts, dialog_lens, s_next_lens, max_len = [], [], [], [], 0
            for i, j in zip(indexes, neg_idx):
                dialog = self.encode_many(self.data[np.arange(i, i+dialog_len+1)])
                s_next = self.encode(self.data[j])
                k = max(len(dialog), len(s_next))
                max_len = k if k > max_len else max_len
                dialogs.append(dialog)
                s_nexts.append(s_next)
                dialog_lens.append(len(dialog))
                s_next_lens.append(len(s_next))
            dialogs = [self.pad_sequence(dialog, max_len) for dialog in dialogs]
            s_nexts = [self.pad_sequence(s_next, max_len) for s_next in s_nexts]
            yield (dialogs, dialog_lens), (s_nexts, s_next_lens)

    def pad_sequence(self, s, max_len, pad=1, return_len=False):
        l = len(s)
        if max_len < l:
            print('Error: padding to %d but seq_len is %d' % (max_len, l))
        if return_len:
            return s + [pad] * (max_len - l), l
        else:
            return s + [pad] * (max_len - l)

    def encode_many(self, ss):
        return [c for s in ss for c in self.encode(s)]

    def encode(self, s):
        valid = self.filt(s, only_seen=True)
        indeces = map(lambda c: self.dict[c], valid)
        return list(indeces)

    def decode(self, m):
        m = filter(lambda n: n >= 0 and n < self.vsize, m)
        return ''.join([self.chars[n] for n in m])
    
    def save(self, fname):
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def filt(self, s, only_seen=False):
        if only_seen:
            s = filter(lambda c: c in self.charset, s)
        else:
            s = re.sub(not_chinese, '', s)
        if self.use_stop_words:
            s = ''.join(filter(lambda c: c not in stop_words, s))
        else:
            s = ''.join(s)
        return s


def load(path):
    try:
        d = Dataset.load(path)
        print('Loaded from ' + path)
    except:
        d = Dataset()
        d.save(path)
    return d
