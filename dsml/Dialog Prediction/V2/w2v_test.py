from utils import *
import jieba, os

path = '~/.datasets/word2vec/cna.cbow.cwe_p.tar_g.512d.0.txt'
w2v = Word2Vec(os.path.expanduser(path))

def w2v_test(dialog, options):
    global w2v

    def sentence_vector(s):
        a = w2v.many_vec(cut_sentence(chinese_only(s), False)).mean(0)
        b = w2v.many_vec(cut_sentence(chinese_only(s), True)).mean(0)
        return (a + b) / 2
    
    x = [sentence_vector(sentence) for sentence in dialog]
    dialog_vector = np.mean(x, 0)
    option_vectors = [sentence_vector(option) for option in options]
    scores = [dialog_vector.dot(ov) for ov in option_vectors]

    try:
        max_opt = np.argmax(scores)
        return max_opt
    except:
        import code; code.interact(local=locals())

do_test(w2v_test)
