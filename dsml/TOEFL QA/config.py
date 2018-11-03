import torch

if torch.cuda.is_available():
    print('Using CUDA')
    cuda = True
else:
    print('Using CPU')
    cuda = False

n_words = 7000
wordvec_path = 'wordvec/wiki-news-300d-20T.vec'

