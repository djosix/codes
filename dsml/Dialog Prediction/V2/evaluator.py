import torch as T
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import numpy as np
from itertools import count
import os


SAVE = 'evaluator.cpt'
CUDA = True
LOSS = 'KLD'

#######################################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(512, 256, 2,
                          dropout=0.2,
                          bidirectional=True)

    def forward(self, x, l):
        # [ batch, step, dim ]
        x, r = self.sort_batch(x, l)
        x, h = self.gru(x, None)
        x = self.unsort_batch(x, r)
        x = x[:, 256:] + x[:, :256]
        x = (x + 1) / 2
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





class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.fc0 = nn.Linear(256, 512)
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.selu(self.fc0(x))
        x = self.drop(x)
        x = F.sigmoid(self.fc1(x))
        return x


#######################################################




def save_model(encoder,
               evaluator,
               optimizer,
               verbose=True,
               name=SAVE):
    data = {
        'encoder': encoder.state_dict(),
        'evaluator': evaluator.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    T.save(data, name)
    if verbose:
        print('model is saved')


def load_model(name=SAVE):
    encoder = Encoder()
    evaluator = Evaluator()
    optimizer = T.optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': evaluator.parameters(), 'lr': 1e-3}
    ])

    if os.path.isfile(name):
        print('model is loaded from file')
        data = T.load(name)
        encoder.load_state_dict(data['encoder'])
        evaluator.load_state_dict(data['evaluator'])
        optimizer.load_state_dict(data['optimizer'])
    
    if CUDA:
        encoder.cuda()
        evaluator.cuda()
    
    return encoder, evaluator, optimizer




#######################################################




if __name__ == '__main__':
    from utils import *

    path = 'cna.cbow.cwe_p.tar_g.512d.0.txt'
    w2v = Word2Vec(os.path.expanduser(path))

    path = 'data/subtitle_no_TC'
    training_set = Dataset(path)

    encoder, evaluator, optimizer = load_model()

    encoder.train()
    evaluator.train()

    try:
        print("=" * 30)
        batch_size = 128
        acc_list = []
        loss_list = []

        for i in count():
            for j in range(training_set.data.shape[0] // batch_size):
                (
                    (dlgs, dlg_lens),
                    (poss, pos_lens),
                    (negs, neg_lens),
                    max_len
                ) = training_set.batch(batch_size, [1, 2, 3], w2v)

                dlgs = V(T.FloatTensor(dlgs))
                poss = V(T.FloatTensor(poss))
                negs = V(T.FloatTensor(negs))

                if CUDA:
                    dlgs = dlgs.cuda()
                    poss = poss.cuda()
                    negs = negs.cuda()

                optimizer.zero_grad()

                # encode inputs
                x = T.cat([dlgs, poss, negs], 0)
                l = [*dlg_lens, *pos_lens, *neg_lens]
                encoded = encoder(x, l)
                dlgs = encoded[0:batch_size]
                poss = encoded[batch_size:batch_size*2]
                negs = encoded[batch_size*2:]

                # make prediction
                pred = evaluator(dlgs)

                # compute loss
                if LOSS == 'KLD':
                    loss_pos = F.kl_div(pred, poss)
                    loss_neg = F.kl_div(pred, negs)
                elif LOSS == 'BCE':
                    loss_pos = F.binary_cross_entropy(pred, poss)
                    loss_neg = F.binary_cross_entropy(pred, negs)
                
                # evaluate accuracy over the mini-batch
                diff_pos = pred - poss
                diff_neg = pred - negs
                score_neg = -(diff_neg * diff_neg).sum(1)
                score_pos = -(diff_pos * diff_pos).sum(1)
                n_correct = (score_pos > score_neg).sum()
                acc = n_correct.data[0] / batch_size
                acc_list += [acc]

                # backpropagate gradient
                loss = loss_pos - loss_neg
                loss.backward()
                nn.utils.clip_grad_norm(encoder.parameters(), 50.0)
                nn.utils.clip_grad_norm(evaluator.parameters(), 50.0)
                loss_list += [loss.data[0]]

                # update
                optimizer.step()

                if j % 100 == 99:
                    print('\rEp:%d[%d] Accuracy:%.2f Loss:%.5f          '
                        % (i, j, np.mean(acc_list), np.mean(loss_list)))
                    save_model(encoder, evaluator, optimizer, False)
                    acc_list.clear()
                    loss_list.clear()
                else:
                    print('\rEp:%d[%d] A:%.2f L+:%.2f L-:%.2f L:%.5f         \r'
                        % (i, j, acc, loss_pos.data[0], loss_neg.data[0], loss.data[0]),
                        end='', flush=True)
                    
            print('\rEpoch %d: Accuracy=%.2f Loss=%.5f'
                    % (i, np.mean(acc_list), np.mean(loss_list)))

    except KeyboardInterrupt:
        save_model(encoder, evaluator, optimizer)

