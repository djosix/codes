import torch as T
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import numpy as np

CUDA = True

class Encoder(nn.Module):
    def __init__(self, num_vocab, dim_embedded, dim_encoded):
        super(Encoder, self).__init__()
        self.num_vocab = num_vocab
        self.dim_embedded = dim_embedded
        self.dim_encoded = dim_encoded
        
        self.embed = nn.Embedding(num_vocab, dim_embedded)
        self.lstm = nn.LSTM(dim_embedded, dim_encoded, 4)
        self.fc1 = nn.Linear(dim_encoded, dim_encoded)
        self.fc2 = nn.Linear(dim_encoded, dim_encoded)
        self.fc3 = nn.Linear(dim_encoded, dim_encoded)

    def forward(self, seq, seq_lens):
        """ NOTE
        Args:
            seq: [n_step, n_batch], LongTensor Variable
            seq_len: [n_batch], int list
        Return:
            [n_step, n_batch, dim]
        ** SORT SEQUENCES WITH THEIR LENGTHS HERE **
        """
        
        seq_lens, i, r = self.sort_seq(seq_lens) # sort lengths
        seq = seq[:, i] # sort sequences lengths: [n_step, n_batch]
        x = self.embed(seq)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens)
        x, h = self.lstm(x, None)
        x, l = nn.utils.rnn.pad_packed_sequence(x)
        i, j = zip(*map(lambda x: (x[0], x[1] - 1), enumerate(l)))
        x = F.relu(x[j, i, :])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = x[r] # unsort sequences by lengths: [n_batch, dim]
        return x
    
    def sort_seq(self, seq_lens):
        seq_lens = T.LongTensor(seq_lens)
        seq_lens = seq_lens.cuda() if CUDA else seq_lens
        sorted_lens, i = seq_lens.sort(0, True)
        r = i.sort()[1]
        return sorted_lens.tolist(), i, r

class Predictor(nn.Module):
    def __init__(self, encoder):
        super(Predictor, self).__init__()
        self.encoder = encoder
        dim = encoder.dim_encoded

        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, dim)

        self.optimizer = T.optim.Adam(self.parameters())
        
    def forward(self, seq, seq_lens):
        x = self.encoder(seq, seq_lens)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
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
    
    def loss(self, x, x_lens, y, y_lens):
        p = self(x, x_lens)
        q = self.encoder(y, y_lens)
        loss = (- q).abs().mean()
        # loss = F.binary_cross_entropy(p, q)
        return loss

    def score(self, x, x_lens, y, y_lens):
        self.eval()
        loss = self.loss(x, x_lens, y, y_lens).data[0]
        return -loss

def model():
    import dataset, os
    d = dataset.load('.ai_dataset')
    encoder = Encoder(d.vsize, 1024, 128)
    m = Predictor(encoder)
    if CUDA:
        m.cuda()
    if os.path.isfile('model.save'):
        data = T.load('model.save')
        m.load_state_dict(data['m'])
        m.optimizer.load_state_dict(data['optimizer'])
        print('Model loaded from file')
    return d, m

if __name__ == '__main__':
    import pickle, os

    d, m = model()

    batches = [
        (d.random_batch_generator(512, 1), d.random_batch_generator_neg(512, 1)),
        (d.random_batch_generator(128, 3), d.random_batch_generator_neg(128, 3)),
        (d.random_batch_generator(256, 2), d.random_batch_generator_neg(256, 2))
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

            print('[%d] %.4e' % (n, loss))

        if n % 10 == 0:
            print('Saving model')
            T.save({'m': m.state_dict(),
                    'optimizer': m.optimizer.state_dict()},
                    'model.save')


    # from test import test
    # test(m, d)
