# import pickle, functools, numpy, jieba
# from scipy.spatial import distance
# labeled, w2v = pickle.load(open('model.pkl', 'rb'))

# @functools.lru_cache(maxsize=50000)
# def name2vec(name):
#     global w2v
#     words = []
#     for word in name.split():
#         words += list(jieba.cut(word))
#     try:
#         vecs = w2v.many_vec(words)
#         if vecs.shape[0] == 0:
#             raise Exception
#         vec = vecs.mean(0)
#     except:
#         vec = numpy.zeros([w2v.dim])
#     return vec


# @functools.lru_cache(maxsize=1000)
# def getLabel(name):
#     global labeled
#     if name in labeled:
#         return labeled[name]
#     nvec = name2vec(name)
#     label_max, sim_max = None, -numpy.inf
#     for item, label in labeled.items():
#         ivec = name2vec(item)
#         sim = distance.cosine(ivec, nvec)
#         if sim > sim_max:
#             label_max, sim_max = label, sim
#     return label_max

import pickle, functools, numpy as np, jieba
from scipy.spatial import distance

try:
    labeled, w2v

except:
    labeled, w2v = pickle.load(open('model.pkl', 'rb'))

@functools.lru_cache(maxsize=50000)
def name2vec(name):
    global w2v
    words = []
    for word in name.split():
        words += list(jieba.cut(word))
    try:
        vecs = w2v.many_vec(words)
        if vecs.shape[0] == 0:
            raise Exception
        vec = vecs.mean(0)
    except:
        vec = np.ones([w2v.dim])
    return vec

try:
    labels, lvecs
except:
    labels = list(set(labeled.values()))
    lvecs = {}
    for label in labels:
        vecs = []
        for item, label_ in labeled.items():
            if label == label_:
                vecs.append(name2vec(item))
        lvecs[label] = np.product(vecs, 0)


@functools.lru_cache(maxsize=1000)
def getLabel(name):
    global labeled
    if name in labeled:
        return labeled[name]
    nvec = name2vec(name)
    label_max, sim_max = None, -np.inf
    for label, lvec in lvecs.items():
        # sim = -distance.cosine(lvec, nvec)
        sim = -((lvec - nvec) ** 2).sum()
        if sim > sim_max:
            label_max, sim_max = label, sim
    return label_max
    





