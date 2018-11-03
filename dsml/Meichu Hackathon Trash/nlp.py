import pickle, tqdm, numpy as np, pickle

try:
    data = pickle.load(open('data.pkl', 'rb'))

except:
    exit()

data = [data[i] for i in data]

iset = set()
for items in tqdm.tqdm(data, 'Build word set'):
    for item in items:
        name = item[0]
        iset.add(name)


import jieba

# corpus = []
# wset = set()
# for item in iset:
#     words = []
#     for word in item.split():
#         words = list(jieba.cut(word))
#     corpus.append(words)
#     wset |= set(words)

# import gensim
# model = gensim.models.Word2Vec(corpus, size=100, window=3, workers=8)


from w2v import Word2Vec
print('Loading Word2Vec...')
wemb = Word2Vec('/Users/dorian/Documents/Datasets/word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt')




try:
    wvec, wvec_all = pickle.load(open('wvec.pkl', 'rb'))
    print('`wvec` is loaded')

except:
    wvec = {}
    wvec_all = []
    for item in tqdm.tqdm(iset, desc='Cutting names'):
        words = []
        for word in item.split():
            words += list(jieba.cut(word))
        try:
            vec = wemb.many_vec(words)
            if vec.shape[0] == 0:
                raise Exception
            wvec[item] = vec.mean(0)
            wvec_all.append(wvec[item])
            
        except:
            wvec[item] = np.ones([wemb.dim])




try:
    labels = pickle.load(open('labels.pkl', 'rb'))
    print('`labels` is loaded...')

except:
    from sklearn.mixture import GaussianMixture

    print('Training GMM...')
    gmm = GaussianMixture(100, verbose=2, verbose_interval=1)
    gmm.fit(wvec_all)

    labels = {}
    for item, vec in tqdm.tqdm(wvec.items(), desc='Clustering'):
        labels[item] = gmm.predict([vec])


clusters = {}
for item, label in labels.items():
    clusters.setdefault(label[0], []).append(item)

pickle.dump(clusters, open('clusters.pkl', 'wb'))
