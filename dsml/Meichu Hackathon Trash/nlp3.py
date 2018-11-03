import pickle, tqdm, numpy as np, pickle

try:
    labeled = pickle.load(open('labeled.pkl', 'rb'))

except:
    exit()



from w2v import Word2Vec
print('Loading Word2Vec...')
wemb = Word2Vec('/Users/dorian/Documents/Datasets/word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt')



import jieba

wvecs = {}
for item in tqdm.tqdm(labeled.keys(), desc='Computing name vectors'):
    words = []
    for word in item.split():
        words += list(jieba.cut(word))
    try:
        vec = wemb.many_vec(words)
        if vec.shape[0] == 0:
            raise Exception
        wvecs[item] = vec.mean(0)
    except:
        wvecs[item] = np.zeros([wemb.dim])



# label : label_id
llist = list(set(labeled.values()) - {False})

means = []
for i, label in enumerate(llist):
    vecs = []
    for item, vec in wvecs.items():
        mark = labeled[item]
        if mark and mark == label:
            vecs.append(vec)
    means.append(np.array(vecs).mean(0))




try:
    gmm = pickle.load(open('gmm3.pkl', 'rb'))

except:
    from sklearn.mixture import GaussianMixture

    print('Training GMM...')
    gmm = GaussianMixture(len(llist), verbose=2, verbose_interval=1, means_init=means)
    gmm.fit(list(wvecs.values()))
    pickle.dump(open('gmm3.pkl', 'wb'))


print('Clustering')
labels = gmm.predict(list(wvecs.values()))
for item, lindex in tqdm.tqdm(zip(wvecs.keys(), labels), desc='Writing labels'):
    if not labeled[item]:
        labeled[item] = llist[lindex]

pickle.dump(labeled, open('final_labels.pkl', 'wb'))

pickle.dump((
    labeled,
    gmm,
    llist,
    wemb
), open('model.pkl', 'wb'))

# import pickle
# from functools import lru_cache

# (
#     labeled,
#     gmm,
#     llist,
#     wemb
# ) = pickle.load(open('model.pkl', 'rb'))


# ''' Use this !!! '''
# @lru_cache(maxsize=200)
# def getLabel(name):
#     global gmm, wemb, labeled, llist
#     if name in labeled:
#         return labeled[name]
#     words = []
#     for word in name.split():
#         words += list(jieba.cut(word))
#     try:
#         vecs = wemb.many_vec(words)
#         if vecs.shape[0] == 0:
#             raise Exception
#         vec = vecs.mean(0)
#     except:
#         vec = np.zeros([wemb.dim])
    
#     index = gmm.predict([vec])[0]
#     return llist[index]
