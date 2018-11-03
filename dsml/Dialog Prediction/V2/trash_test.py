import torch as T
from torch.autograd import Variable as V
from pytorch_gpu import model
import os, numpy as np

d, m = model()

def get_problems(fname):
    with open(fname, 'rt') as f:
        content = f.read()
    data = {}
    for line in content.split('\n')[1:-1]:
        no, dialog, options = line.split(',')
        data[int(no)] = {
            'dialog': dialog.split('\t'),
            'options': options.split('\t')
        }
    return data

problems = get_problems('AIFirstProblem.txt')
result = []
for i in problems:
    dialog = problems[i]['dialog']
    dialog = d.encode_many(dialog)
    max_opt, max_val = -1, -np.inf
    print('Question %d' % i)
    print('    ' + '\n    '.join(problems[i]['dialog']))
    for j, option in enumerate(problems[i]['options']):
        option = d.encode(option)
        padding = max(len(dialog), len(option))
        s1 = d.pad_sequence(dialog, padding)
        s2 = d.pad_sequence(option, padding)
        s1 = V(T.LongTensor(s1)).view(-1, 1).cuda()
        s2 = V(T.LongTensor(s2)).view(-1, 1).cuda()
        k = m.score(s1, [len(dialog)], s2, [len(option)])
        print('(%d)' % j, int(np.exp(k) * 1e10), problems[i]['options'][j])
        if k > max_val:
            max_opt, max_val = j, k
    print('===> (%d)' % max_opt, problems[i]['options'][max_opt])
    result += [(i, max_opt)]
with open('result.csv', 'wt') as f:
    a = ['%d,%d' % (i, j) for i, j in result]
    a = ['id,ans'] + a
    s = '\n'.join(a)
    f.write(s)
