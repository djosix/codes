import jieba, pickle, tqdm, os, re
import numpy as np

################################################################
################################################################
################################################################

def chinese_only(sentence):
    return re.sub('[^\u4e00-\u9fff]', '', sentence)

def cut_sentence(sentence, cut_all=True):
    return list(jieba.cut(sentence, cut_all=cut_all))

# requires a 2d array
def pad0(sequence, dim0):
    size = (dim0, sequence.shape[1])
    return np.resize(sequence, size)

################################################################
################################################################
################################################################

class Dataset:

    def __init__(self, root):
        self.root = root.rstrip('/')
        self.pkl = root + '.cache'

        try:
            self._load_pkl()
            print('dataset is loaded from cache')
        except:
            self._load_data()
            self._save_pkl()



    def _load_data(self):
        self.data = []
        self.max_len = 0

        # iterate over all files
        for path, dirs, files in os.walk(self.root):
            files = tqdm.tqdm(files, desc=path)
            for f in files:
                if f.endswith('.txt'):
                    self._load_file(os.path.join(path, f))

        self.data = np.array(self.data)



    def _load_file(self, path):
        with open(path, 'rt') as f:

            # filter non chinese words and cut with jieba
            lines = [self.preprocess(s) for s in f.readlines()]
        
        max_len = max(len(s) for s in lines)
        if max_len > self.max_len:
            self.max_len = max_len
        
        self.data += lines
        


    def _load_pkl(self):
        with open(self.pkl, 'rb') as f:
            data = pickle.load(f)
        self.data, self.max_len = data
    


    def _save_pkl(self):
        data = self.data, self.max_len
        with open(self.pkl, 'wb') as f:
            pickle.dump(data, f)


    def preprocess(self, sentence):
        return cut_sentence(chinese_only(sentence))


    def batch(self, batch_size, dialog_sizes, w2v):
        
        data_size = self.data.shape[0]
        dialog_max_index = data_size - max(dialog_sizes) - 1

        # dialog indeces
        indexes = np.random.randint(dialog_max_index, size=batch_size)

        dlgs, dlg_lens = [], []
        poss, pos_lens = [], []
        negs, neg_lens = [], []
        max_len = 0

        for i in indexes:
            
            while True:
                offset = np.random.choice(dialog_sizes)
                dlg = np.concatenate(self.data[i:i+offset])
                dlg = w2v.many_vec(dlg)

                pos = self.data[i+offset]
                pos = w2v.many_vec(pos)

                if len(dlg.shape) + len(pos.shape) < 4:
                    i = np.random.randint(dialog_max_index)
                else:
                    break

            while True:
                neg = self.data[np.random.randint(data_size)]
                neg = w2v.many_vec(neg)

                if len(neg.shape) < 2:
                    i = np.random.randint(data_size)
                else:
                    break

            # feature vectors
            dlgs += [dlg]
            poss += [pos]
            negs += [neg]

            # sequence lengths
            dlg_lens += [len(dlg)]
            pos_lens += [len(pos)]
            neg_lens += [len(neg)]

            # max length
            m = max(len(dlg), len(pos), len(neg))
            max_len = m if m > max_len else max_len
        
        dlgs = np.array([pad0(s, max_len) for s in dlgs])
        poss = np.array([pad0(s, max_len) for s in poss])
        negs = np.array([pad0(s, max_len) for s in negs])

        batch = [
            (dlgs, dlg_lens),
            (poss, pos_lens),
            (negs, neg_lens),
            max_len
        ]

        return batch


    def batches(self, batch_size, dialog_size, w2v):
        data_size = self.data.shape[0]
        dialog_max_index = data_size - dialog_size - 1
        rand_index = np.random.permutation(dialog_max_index)
        num_batches = data_size // batch_size

        for i in range(data_size // batch_size):

            indexes = rand_index[i:i+batch_size]

            dlgs, dlg_lens = [], []
            poss, pos_lens = [], []
            negs, neg_lens = [], []
            max_len = 0

            for i in indexes:
                
                while True:
                    dlg = np.concatenate(self.data[i:i+dialog_size])
                    dlg = w2v.many_vec(dlg)

                    pos = self.data[i+dialog_size]
                    pos = w2v.many_vec(pos)

                    if len(dlg.shape) + len(pos.shape) < 4:
                        i = np.random.randint(dialog_max_index)
                    else:
                        break

                while True:
                    neg = self.data[np.random.randint(data_size)]
                    neg = w2v.many_vec(neg)

                    if len(neg.shape) < 2:
                        i = np.random.randint(data_size)
                    else:
                        break

                # feature vectors
                dlgs += [dlg]
                poss += [pos]
                negs += [neg]

                # sequence lengths
                dlg_lens += [len(dlg)]
                pos_lens += [len(pos)]
                neg_lens += [len(neg)]

                # max length
                m = max(len(dlg), len(pos), len(neg))
                max_len = m if m > max_len else max_len
            
            dlgs = np.array([pad0(s, max_len) for s in dlgs])
            poss = np.array([pad0(s, max_len) for s in poss])
            negs = np.array([pad0(s, max_len) for s in negs])

            batch = [
                (dlgs, dlg_lens),
                (poss, pos_lens),
                (negs, neg_lens),
                max_len
            ]

            yield batch



################################################################
################################################################
################################################################

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

################################################################
################################################################
################################################################

def do_test(func, fname='data/AIFirstProblem.txt', output='result.csv'):
    # read file
    with open(fname, 'rt') as f:
        content = f.read()

    # parse questions
    questions = {}
    for line in content.split('\n')[1:-1]:
        no, dialog, options = line.split(',')
        questions[int(no)] = {
            'dialog': dialog.split('\t'),
            'options': options.split('\t')
        }

    # inference
    result = []
    for i in questions:
        dialog = questions[i]['dialog']
        options = questions[i]['options']
        print('Question %d' % i)
        print('    ' + '\n    '.join(dialog))
        max_opt = func(dialog, options) # list, list
        print('===> (%d)' % max_opt, options[max_opt])
        result += [(i, max_opt)]
    
    # save result
    with open('result.csv', 'wt') as f:
        a = ['%d,%d' % (i, j) for i, j in result]
        a = ['id,ans'] + a
        s = '\n'.join(a)
        f.write(s)

################################################################
################################################################
################################################################

if __name__ == '__main__':
    print('Hello')

    path = '~/.datasets/word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt'
    w2v = Word2Vec(os.path.expanduser(path))

    path = 'data/subtitle_no_TC'
    d = Dataset(path)
