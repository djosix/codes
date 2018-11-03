import torch as T
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import numpy as np

CUDA = True

class Model(nn.Module):
    def __init__(self,
                 num_vocab,
                 dim_embed,
                 max_seq_len,
                 dim_gru_hidden,
                 num_gru_layers,
                 fc_units):
        super(Model, self).__init__()
        self.num_vocab = num_vocab
        self.dim_embed = dim_embed
        self.max_seq_len = max_seq_len
        self.dim_gru_hidden = dim_gru_hidden
        self.num_gru_layers = num_gru_layers
        self.fc_units = fc_units
        
        self.embed = nn.Embedding(num_vocab, dim_embed)#, max_seq_len)
        self.gru = nn.GRU(dim_embed, dim_gru_hidden, num_gru_layers)
        in_dim = dim_gru_hidden
        self.fc = []
        for out_dim in fc_units:
            self.fc += [nn.Linear(in_dim, out_dim)]
            in_dim = out_dim
        self.fc_end = nn.Linear(in_dim, dim_embed)

        self.opt = T.optim.RMSprop(self.parameters(), lr=0.01)

    def forward(self, x, seq_lens):
        x = F.relu(self.encode_sentence(x, seq_lens))
        for fc in self.fc:
            x = F.relu(fc(x))
        x = F.tanh(self.fc_end(x))
        return x

    def encode_sentence(self, x, seq_lens):
        # x [n_batch, n_word]
        padded = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lens)
        outputs, hidden = self.gru(packed, None)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs)
        i, j = zip(*map(lambda x: (x[0], x[1] - 1), enumerate(output_lengths)))
        return outputs[j, i, :] # [step, batch, dim]

    def predict(self, x, seq_lens):
        self.eval()
        return self.forward(x, seq_lens)

    def fit(self, x, x_seq_lens, x_target, x_target_len):
        self.train()
        self.zero_grad()
        h = self.forward(x, x_seq_lens)
        y = self.encode_sentence(x_target, x_target_len)
        loss = ((y - h) ** 2).mean()
        loss.backward()
        self.opt.step()
        return loss

    def score(self, x, x_seq_lens, x_target, x_target_len):
        self.eval()
        h = self.forward(x, x_seq_lens)
        y = self.encode_sentence(x_target, x_target_len)
        loss = ((y - h) ** 2).mean()
        return (1 / loss).data[0]


def test(m):
    from test import get_problems
    result = []
    data = get_problems('AIFirstProblem.txt')
    for i in data:
        s1, l1 = d.encode_with_padding(data[i]['dialog'][-1])
        s1 = V(T.LongTensor(s1)).view(-1, 1)
        l1 = [l1]
        max_opt, max_val = -1, -np.inf
        for j, option in enumerate(data[i]['options']):
            s2, l2 = d.encode_with_padding(option)
            s2 = V(T.LongTensor(s2)).view(-1, 1)
            l2 = [l2]
            k = m.score(s1, l1, s2, l2)
            if k > max_val:
                max_opt, max_val = j, k
        print(i, data[i]['dialog'][-1], data[i]['options'][max_opt])
        result += [(i, max_opt)]
    with open('result.csv', 'wt') as f:
        a = ['%d,%d' % (i, j) for i, j in result]
        a = ['id,ans'] + a
        s = '\n'.join(a)
        f.write(s)
    return result



if __name__ == '__main__':
    import dataset
    d = dataset.load('.ai_dataset')
    
    m = Model(num_vocab=d.vsize,
              dim_embed=100,
              max_seq_len=d.max_seq_len,
              dim_gru_hidden=100,
              num_gru_layers=5,
              fc_units=[100, 100, 100])

    def n_batch(n):
        g = d.dialog_and_next(1, d.max_seq_len)
        while True:
            x1, x2 = [], []
            for i in range(n):
                (s1, s2), (l1, l2) = next(g)
                x1 += [(s1, l1)]
                x2 += [(s2, l2)]
            x1.sort(key=lambda i: i[1], reverse=True)
            x2.sort(key=lambda i: i[1], reverse=True)
            s1, l1 = zip(*x1)
            s2, l2 = zip(*x2)
            s1 = np.array(s1).T
            s2 = np.array(s2).T
            yield s1, l1, s2, l2

    batches = n_batch(1000)
    for s1, l1, s2, l2 in batches:
        s1 = V(T.from_numpy(s1))
        s2 = V(T.from_numpy(s2))
        loss = m.fit(s1, l1, s2, l2)
        print(loss.data[0])
    save_csv(test(m))
