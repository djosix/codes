import torch as T
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import numpy as np
from itertools import count
import os

SAVE = 'encoder.cpt'
CUDA = True


#######################################################



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(512, 1024, 2,
                          dropout=0.1)

    def forward(self, x, l):
        # [ batch, step, dim ]
        x, r = self.sort_batch(x, l)
        x, h = self.lstm(x, None)
        x = self.unsort_batch(x, r)
        # [ batch, dim ]
        return x

    def sort_batch(self, x, l):
        l = T.LongTensor(l)
        l = l.cuda() if CUDA else l
        l, i = l.sort(0, True)
        r = i.sort()[1]
        x = nn.utils.rnn.pack_padded_sequence(x[i],
                l.tolist(), batch_first=True)
        return x, r

    def unsort_batch(self, x, r):
        x, l = nn.utils.rnn.pad_packed_sequence(x)
        f = lambda x: (x[0], x[1] - 1)
        i, j = zip(*map(f, enumerate(l)))
        x = x[j, i][r]
        return x





#######################################################



def save_model(encoder, optimizer, verbose=True):
    data = {
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    T.save(data, SAVE)
    if verbose:
        print('model is saved')


def load_model():
    encoder = Encoder()
    optimizer = T.optim.Adam(encoder.parameters())

    if os.path.isfile(SAVE):
        print('model is loaded from file')
        data = T.load(SAVE)
        encoder.load_state_dict(data['encoder'])
        optimizer.load_state_dict(data['optimizer'])
    
    if CUDA:
        encoder.cuda()
    
    return encoder, optimizer




#######################################################




if __name__ == '__main__':
    from utils import *

    path = 'cna.cbow.cwe_p.tar_g.512d.0.txt'
    w2v = Word2Vec(os.path.expanduser(path))

    path = 'data/subtitle_no_TC'
    training_set = Dataset(path)

    encoder, optimizer = load_model()

    encoder.train()


    try:
        print("=" * 30)
        batch_size = 128
        acc_list = []

        for i in count():
            for j in range(100):
                n = np.random.choice([1, 2, 3])
                (
                    (dlgs, dlg_lens),
                    (poss, pos_lens),
                    (negs, neg_lens),
                    max_len
                ) = training_set.batch(batch_size, n, w2v)

                dlgs = V(T.FloatTensor(dlgs))
                poss = V(T.FloatTensor(poss))
                negs = V(T.FloatTensor(negs))

                if CUDA:
                    dlgs = dlgs.cuda()
                    poss = poss.cuda()
                    negs = negs.cuda()

                encoder.zero_grad()

                x = T.cat([dlgs, poss, negs], 0)
                l = [*dlg_lens, *pos_lens, *neg_lens]

                encoded = encoder(x, l)

                dlgs = encoded[0:batch_size]
                poss = encoded[batch_size:batch_size*2]
                negs = encoded[batch_size*2:]

                assert dlgs.size() == poss.size() == negs.size()

                score_pos = T.mean(dlgs * poss, 1)
                score_neg = T.mean(dlgs * negs, 1)
                assert score_pos.size() == T.Size([batch_size])

                n_correct = (score_pos > score_neg).cpu().data.numpy().sum()
                acc = n_correct / batch_size
                acc_list += [acc]

                neg_sum = score_neg.sum()
                pos_sum = score_pos.sum()
                loss = neg_sum - pos_sum

                # print(acc, score_pos.data[0], score_neg.data[0], loss.data[0])
                print('\r%d %.2f' % (j, acc), end='', flush=True)

                loss.backward()
                nn.utils.clip_grad_norm(encoder.parameters(), 50.0)

                optimizer.step()

            
            print('\repoch %d: accuracy=%.2f' % (i, np.mean(acc_list)))
            acc_list = []
            save_model(encoder, optimizer, verbose=False)

    except KeyboardInterrupt:
        save_model(encoder, optimizer)
