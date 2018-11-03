from utils import *
import jieba, os
from evaluator import *

import torch as T
from torch.autograd import Variable as V

path = 'cna.cbow.cwe_p.tar_g.512d.0.txt'
w2v = Word2Vec(os.path.expanduser(path))

encoder, evaluator, optimizer = load_model()

encoder.eval()
evaluator.eval()

def test(dialog, options):
    global w2v

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

    encoded = encoder(x, l)
    
    dlg = encoded[0:1]
    pred = evaluator(dlg)
    opts = encoded[1:]
    if LOSS == 'KLD':
        loss = [F.kl_div(pred, opt).mean().data[0]
                for opt in opts]
    elif LOSS == 'BCE':
        loss = [F.binary_cross_entropy(pred, opt).mean().data[0]
                for opt in opts]

    max_opt = np.argmin(loss)
    return max_opt

    try:
        pass
    except Exception as ex:
        print(ex)
        import code; code.interact(local=locals())

do_test(test)
