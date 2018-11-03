import torch as T
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import numpy as np

CUDA = True
SAVE = 'trash.cpt'

class Encoder(nn.Module):
    def __init__(self,
                 num_vocab,
                 dim_embedded,
                 dim_encoded):
        super(Encoder, self).__init__()
        self.dim_encoded = dim_encoded
        self.dim_embedded = dim_embedded
        
        self.embed = nn.Embedding(num_vocab, dim_embedded)
        self.gru = nn.GRU(dim_embedded, dim_encoded, 1,
                          dropout=0.5, bidirectional=True)

    def forward(self, x, l):
        """ NOTE
        Args:
            seq: [n_step, n_batch], LongTensor Variable
            seq_len: [n_batch], int list
        Return:
            [n_step, n_batch, dim]
        ** SORT SEQUENCES WITH THEIR LENGTHS HERE **
        """
        
        x = self.embed(x)
        x, l = self.sort_batch(x)
        x, h = self.gru(x, None)
        x = self.unsort_batch(x)
        x = (x[:self.dim_ecoded] + x[self.dim_encoded:]) / 2
        x = (x + 1) / 2
        return x
    
    def sort_batch(self, x, l):
        l = T.LongTensor(l)
        l = l.cuda() if CUDA else l
        l, i = l.sort(0, True)
        self.r = i.sort()[1]
        x = nn.utils.rnn.pack_padded_sequence(x[i],
                l.tolist())#, batch_first=True)
        return x

    def unsort_batch(self, x):
        x, l = nn.utils.rnn.pad_packed_sequence(x)
        f = lambda x: (x[0], x[1] - 1)
        i, j = zip(*map(f, enumerate(l)))
        x = x[j, i][self.r]
        return x





class Predictor(nn.Module):
    def __init__(self, num_vocab,
                       dim_embedded,
                       dim_encoded):
        super(Predictor, self).__init__()
        self.encoder = Encoder()
        self.fc1 = nn.Linear(dim_encoded, dim_encoded * 2)
        self.fc2 = nn.Linear(dim_encoded * 2, dim_encoded * 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(dim_encoded * 2, dim_encoded)

        self.optimizer = T.optim.Adam(self.parameters())
        
    def forward(self, x, l):
        x = self.encoder(x, l)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x

    def fit(self, pos_batch, neg_batch):
        self.train()
        self.zero_grad()
        pX, pX_lens, pY, pY_lens = pos_batch
        nX, nX_lens, nY, nY_lens = neg_batch
        loss = self.loss(pX, pX_lens, pY, pY_lens)
        loss -= self.loss(nX, nX_lens, nY, nY_lens)
        loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), 50.0)
        self.optimizer.step()
        return loss.data[0]
    
    def loss(self, x, xl, y, yl):
        p = self(x, xl)
        q = self.encoder(y, yl)
        return F.binary_cross_entropy(p, q)

    def score(self, x, x_lens, y, y_lens):
        self.eval()
        loss = self.loss(x, x_lens, y, y_lens).data[0]
        return -loss


if __name__ == '__main__':
    import dataset, pickle, os
    d = dataset.load('.ai_dataset')
    
    encoder = Encoder(d.vsize, 512, 256)
    m = Predictor(encoder)
    if CUDA:
        m.cuda()

    if os.path.isfile(SAVE):
        data = T.load(SAVE)
        m.load_state_dict(data['m'])
        m.optimizer.load_state_dict(data['optimizer'])
        print('Model loaded from file')

    batches = [
        (d.random_batch_generator(512, 1), d.random_batch_generator_neg(512, 1)),
        (d.random_batch_generator(512, 3), d.random_batch_generator_neg(512, 3)),
        (d.random_batch_generator(512, 2), d.random_batch_generator_neg(512, 2))
    ]

    for n in range(100000):
        for pos_batch, neg_batch in batches:
            (pD, pD_lens), (pS, pS_lens) = next(pos_batch)
            (nD, nD_lens), (nS, nS_lens) = next(neg_batch)

            pD = V(T.LongTensor(pD)).t()
            pS = V(T.LongTensor(pS)).t()
            nD = V(T.LongTensor(nD)).t()
            nS = V(T.LongTensor(nS)).t()

            if CUDA:
                pD = pD.cuda()
                pS = pS.cuda()
                nD = nD.cuda()
                nS = nS.cuda()

            pos_batch = (pD, pD_lens, pS, pS_lens)
            neg_batch = (nD, nD_lens, nS, nS_lens)
            loss = m.fit(pos_batch, neg_batch)

            print('[%d] %.5f' % (n, loss * 1e10))

        if n % 10 == 0:
            print('Saving model')
            T.save({'m': m.state_dict(),
                    'optimizer': m.optimizer.state_dict(),
                    SAVE)


    # from test import test
    # test(m, d)
