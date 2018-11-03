from w2v import Word2Vec
import pickle, jieba, numpy as np, tqdm
from scipy.spatial import distance

wemb = Word2Vec('/Users/dorian/Documents/Datasets/word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt')


data = pickle.load(open('data.pkl', 'rb'))
data = [i[0] for user in data.values() for i in user]
data = list(set(data))
wvec = {}
for name in tqdm.tqdm(data, desc='Computing word vectors'):
    words = []
    for word in name.split():
        words += list(jieba.cut(word))
    vec = wemb.many_vec(words)
    if vec.shape[0] == 0:
        wvec[name] = np.ones([wemb.dim])
    else:
        wvec[name] = vec.mean(0)


luis = pickle.load(open('luis.pkl', 'rb'))
lvecs = {}
for label, words in tqdm.tqdm(luis.items(), desc='Computing label vectors'):
    vecs = []
    for word in words:
        vec = wemb.many_vec(list(jieba.cut(word)))
        if vec.shape[0] != 0:
            vecs.append(vec.mean(0))
    if vecs:
        lvecs[label] = vecs


labeled = {}
for name, vec in tqdm.tqdm(wvec.items(), desc='Labeling'):
    label_max, sim_max = None, -np.inf
    for label, vecs in lvecs.items():
        # sim = np.max([distance.cosine(vec, lvec) for lvec in vecs])
        sim = np.max([((vec - lvec) ** 2).mean() for lvec in vecs])
        if sim > sim_max:
            label_max, sim_max = label, sim
    labeled[name] = label_max