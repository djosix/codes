
import jieba, pickle, tqdm, os, re
import numpy as np

class Word2Vec:
    def __init__(self, fname):
        self.fname = fname
        self.pkl = fname + '.cache'
        try:
            self._load_pkl()
            print('word2vec is loaded from cache')
        except:
            self._load_data()
            self._save_pkl()
    
    def _load_data(self):
        self.dim, self.dict = 0, {}
        with open(self.fname) as f:
            for line in tqdm.tqdm(f, desc='[word2vec]'):
                tokens = line.strip().split()
                if len(tokens) == 2:
                    self.dim = int(tokens[1])
                    continue
                word = tokens[0]
                vec = np.array([float(t) for t in tokens[1:]])
                self.dict[word] = vec

    def _load_pkl(self):
        with open(self.pkl, 'rb') as f:
            data = pickle.load(f)
            self.dim, self.dict = data

    def _save_pkl(self):
        with open(self.pkl, 'wb') as f:
            data = (self.dim, self.dict)
            pickle.dump(data, f)

    def vec(self, word):
        try:
            return self.dict[word]
        except:
            return None
    
    def many_vec(self, words):
        to_vector = lambda word: self.vec(word)
        not_none = lambda item: item is not None
        result = filter(not_none, map(to_vector, words))
        return np.array(list(result))

    def vecs(self, sentence):
        words = cut_sentence(sentence)
        return self.many_vec(words)

