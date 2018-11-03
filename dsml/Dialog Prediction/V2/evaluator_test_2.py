from utils import *
import jieba, os
from evaluator import *

import torch as T
from torch.autograd import Variable as V

path = 'cna.cbow.cwe_p.tar_g.512d.0.txt'
w2v = Word2Vec(os.path.expanduser(path))

models = [load_model('evaluator_%d.cpt' % i) for i in range(9)]

for encoder, evaluator, _ in models:
    encoder.eval()
    evaluator.eval()

def test(dialog, options):

    def fuck(s):
        return w2v.many_vec(cut_sentence(chinese_only(s)))

    dlg = np.concatenate([fuck(s) for s in dialog])
    opts = [fuck(option) for option in options]

    a = [dlg, *opts]

    l = [len(i) for i in a]
    max_len = max(l)

    x = np.array([pad0(i , max_len) for i in a])
    x = V(T.from_numpy(x.astype(np.float32)))
    x = x.cuda() if CUDA else x

    loss = np.zeros(7)
    for encoder, evaluator, _ in models:
        encoded = encoder(x, l)
        
        _dlg = encoded[0:1]
        _pred = evaluator(_dlg)
        _opts = encoded[1:]
        if LOSS == 'KLD':
            _loss = [F.kl_div(_pred, opt).mean().data[0]
                    for opt in _opts]
        elif LOSS == 'BCE':
            _loss = [F.binary_cross_entropy(_pred, opt).mean().data[0]
                    for opt in _opts]

        loss = np.array(_loss)
    return loss.argmin()

    try:
        pass
    except Exception as ex:
        print(ex)
        import code; code.interact(local=locals())

do_test(test)
