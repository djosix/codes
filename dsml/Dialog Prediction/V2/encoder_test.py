from utils import *
import jieba, os
from encoder import *

import torch as T
from torch.autograd import Variable as V


path = 'cna.cbow.cwe_p.tar_g.512d.0.txt'
w2v = Word2Vec(os.path.expanduser(path))

encoder, optimizer = load_model()

encoder.eval()

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
    assert x.shape == (7, max_len, 512)

    x = V(T.from_numpy(x.astype(np.float32)))
    assert x.size() == T.Size([7, max_len, 512])
    assert x.data.type() == 'torch.FloatTensor'
    x = x.cuda() if CUDA else x

    y = encoder(x, l).data
    y = y.cpu() if CUDA else y
    [dlg, *opts] = y.numpy()

    scores = [np.mean(dlg * opt) for opt in opts]

    try:
        max_opt = np.argmax(scores)
        return max_opt
    except Exception as ex:
        print(ex)
        import code; code.interact(local=locals())

do_test(test)
