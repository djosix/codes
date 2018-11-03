from collections import Counter, OrderedDict
import pickle, os, pandas as pd, numpy as np
from nltk.tokenize import word_tokenize
from progress.bar import Bar
from config import defaults
from nltk.stem.snowball import EnglishStemmer



__all__ = ['load_dataset', 'load_embedding', 'Tokenizer']

def load_dataset():
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    with open('cache/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
        train_x = pad_sequences(dataset['train_x'], maxlen=defaults['input_len'])
        train_y = to_categorical(dataset['train_y'])
        test_x = pad_sequences(dataset['test_x'], maxlen=defaults['input_len'])
    return train_x, train_y, test_x


def load_embedding():
    with open('cache/data_emb_init.np', 'rb') as f:
        weights = np.load(f)
    return weights


stop_words = {
    *',.?/:;\\\'"!@#$%^&*()_+=-',
    'a', 'an', 'the', 'am', 'is', 'are', 'do', 'does',
    'did', 'was', 'were', 'has', 'have', 'had', 'be', 'been',
}


class Tokenizer:
    def __init__(self):
        self.word_counts = {
            '<NUL>': 2 ** 64,
            '<OOV>': 2 ** 64 - 1}
        self.stemmer = EnglishStemmer()
    
    def add_texts(self, texts):
        full_text = ' '.join(texts)
        counter = Counter(self.tokenize(full_text))
        for word, count in Bar('adding texts').iter(counter.items()):
            if word in self.word_counts:
                self.word_counts[word] += count
            else:
                self.word_counts[word] = count
    
    def tokenize(self, text):
        text = text.replace('-', ' ')
        for word in word_tokenize(text):
            word = self.stemmer.stem(word)
            if word not in stop_words:
                yield word
    
    def freeze(self, size=None):
        if size is not None:
            counter = Counter(self.word_counts)
            self.word_counts = counter.most_common(size)
        else:
            self.word_counts = list(self.word_counts.items())
        self.word_index = OrderedDict([
            (w, i) for i, (w, _) in enumerate(self.word_counts)])
        del self.word_counts
    
    def text_to_sequence(self, text):
        return [
            self.word_index.get(word, 1)
            for word in self.tokenize(text)]

    def texts_to_sequences_generator(self, texts):
        for text in texts:
            yield self.text_to_sequence(text)
    
    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def _make_tokenizer():
    tokenizer = Tokenizer()
    print('training data')
    train_data = pd.read_csv('./data/train.tsv', delimiter='\t')
    tokenizer.add_texts(train_data['Phrase'])
    print('testing data')
    test_data = pd.read_csv('./data/test.tsv', delimiter='\t')
    tokenizer.add_texts(test_data['Phrase'])
    return tokenizer
    

def _pick_vectors(tokenizer):
    all_words = set(tokenizer.word_counts.keys())
    vectors = {}
    with open('/Users/dorian/Documents/Datasets/crawl-300d-2M.vec') as f:
        n_lines, dim = map(int, f.readline().split())
        for i in Bar('picking vectors').iter(range(n_lines)):
            line = f.readline().split()
            if not all_words or not line:
                break
            word, *vals = line
            if word in all_words:
                vectors[word] = list(map(float, vals))
                all_words.remove(word)
    for word in all_words:
        vectors[word] = [0] * 300
    return vectors


def _make_embed_init(tokenizer, vectors):
    print('making embedding init weights')
    weights = []
    for i, (word, j) in enumerate(tokenizer.word_index.items()):
        assert i == j
        weights.append(vectors[word])
    return np.array(weights)


def _make_dataset(tokenizer):
    print('making dataset')
    train_data = pd.read_csv('./data/train.tsv', delimiter='\t')
    test_data = pd.read_csv('./data/test.tsv', delimiter='\t')
    return {
        'train_x': tokenizer.texts_to_sequences(train_data['Phrase']),
        'train_y': train_data['Sentiment'],
        'test_x': tokenizer.texts_to_sequences(test_data['Phrase'])}


if __name__ == '__main__':
    tokenizer = _make_tokenizer()

    with open('cache/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    vectors = _pick_vectors(tokenizer)
    with open('cache/vectors.pkl', 'wb') as f:
        pickle.dump(vectors, f)

    # Select dictionary size
    tokenizer.freeze(defaults['input_dim'])

    weights = _make_embed_init(tokenizer, vectors)
    with open('cache/data_emb_init.np', 'wb') as f:
        np.save(f, np.array(weights))
    
    dataset = _make_dataset(tokenizer)
    with open('cache/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
