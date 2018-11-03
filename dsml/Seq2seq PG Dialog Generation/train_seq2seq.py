"""

Saving:
checkpoint/*.ckpt

Corpus: 1 sentence per line
data/<name>/**/*.txt

Cache:
cache/*.cache



"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os, tqdm, importlib, random, itertools, time, sys
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored



from rl_chatbot import config, utils, models, data, seq2seq_models
importlib.reload(config)
importlib.reload(utils)
importlib.reload(models)
importlib.reload(data)
importlib.reload(seq2seq_models)


# Save to another file at these epochs
SAVE_POINTS = [1, 20, 50, 100, 200, 300, 500, 890]


try:
    # Get options from command line arguments
    options = sys.argv[1:]
    CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD = options
    HIDDEN_DIMENSION, GRU_LAYERS = int(HIDDEN_DIMENSION), int(GRU_LAYERS)
    TEACHER_FORCING_RATIO, DROPOUT, CLIP_GRAD = float(TEACHER_FORCING_RATIO), float(DROPOUT), float(CLIP_GRAD)
    TOKENIZER = eval(TOKENIZER)
    print('CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD')
    print(CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD)

except:
    # Use default options
    CORPUS = 'chinese'
    TARGET = 'next'
    TOKENIZER = 'utils.jieba_tokenizer'
    TEACHER_FORCING_RATIO = 0.5
    HIDDEN_DIMENSION = 512
    GRU_LAYERS = 1
    DROPOUT = 0.3
    CLIP_GRAD = 10
    options = map(str, [CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD])
    print('CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD')
    TOKENIZER = eval(TOKENIZER) # Get function
    print(CORPUS, TARGET, TOKENIZER, TEACHER_FORCING_RATIO, HIDDEN_DIMENSION, GRU_LAYERS, DROPOUT, CLIP_GRAD)


# Checkpoint name
options = list(options)
LOAD = 'seq2seq_{}'.format('_'.join(options))
SAVE = 'seq2seq_{}'.format('_'.join(options))
print(TARGET, LOAD, '=>', SAVE)


#==================================================
# Load

corpus_options = {
    'name': CORPUS, # data/<name>/**/*.txt
    'dict_size': config.DICT_SIZE,
    'tokenizer': TOKENIZER,
    'target': TARGET, # <self|next|prev>
    'file_filter': lambda name: name.endswith('.txt'),
    'use_cache': True
}

model_options = {
    'n_words': config.DICT_SIZE,
    'hidden_size': HIDDEN_DIMENSION,
    'gru_layers': GRU_LAYERS,
    'dropout': DROPOUT
}

config.MIN_INPUT_LEN = 1
config.MAX_INPUT_LEN = 15
config.MIN_TARGET_LEN = 1
config.MAX_TARGET_LEN = 15
config.MAX_DECODE_LEN = 15

corpus = utils.load_corpus(**corpus_options)
model = seq2seq_models.AttnSeq2Seq(**model_options)
optimizer = optim.Adam(model.parameters(), lr=0.001)

load_path = os.path.join(config.CHECKPOINT_DIR, '{}.ckpt'.format(LOAD))
save_path = os.path.join(config.CHECKPOINT_DIR, '{}.ckpt'.format(SAVE))

if os.path.isfile(load_path):
    print('Load session from {}'.format(load_path))
    states = torch.load(load_path)
    epoch = states['epoch']
    losses = states['losses']
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['optimizer'])

else:
    print('New session')
    epoch = 1
    losses = []

if config.CUDA:
    model.cuda()



#==================================================
# Functions

def predict(sentences):
    index_seqs = corpus.sentences_to_index_seqs(sentences)
    seqs, seq_lens = corpus.index_seqs_variable(index_seqs)
    seqs = seqs.cuda() if config.CUDA else seqs
    outputs = model(seqs, seq_lens)
    corpus.index_seqs_to_sentences(outputs, sep='', token=False)
    return corpus.index_seqs_to_sentences(outputs, sep='', token=False)


def test(inputs, input_lens, sep='', token=False):
    size = min(15, len(input_lens))
    i = sorted(np.random.choice(len(input_lens), size, replace=False).tolist())
    inputs = inputs[:, i]
    input_lens = np.take(input_lens, i).tolist()
    sentences = corpus.index_seqs_to_sentences(inputs.t().data.numpy(), sep=sep)
    outputs = model(inputs.cuda() if config.CUDA else inputs, input_lens).data
    responses = corpus.index_seqs_to_sentences(outputs, sep=sep, token=token)
    for sentence, response in zip(sentences, responses):
        print(sentence, '=>', colored(response, 'magenta', attrs=['bold']))


def plot(losses, group, show=True):
    # os.environ['DISPLAY'] = ':0'
    n = len(losses) // group
    losses = np.array(losses[:n * group]).reshape(n, group).mean(1)
    if show: plt.clf()
    plt.plot(np.arange(losses.shape[0]), losses)
    if show: plt.show()


def save(name):
    print('Saving to {}'.format(name))
    torch.save({
        'epoch': epoch,
        'losses': losses,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, name)


#==================================================
# Training


done = False
for batches in utils.general_batches_generator(corpus, 64, 'replace'):

    # Calculate time
    start_time = time.time()

    # Do a test
    model.eval()
    test(*batches[0][0])    
    model.train()

    L = []
    print('Loss: --')

    try:
        for batch in utils.progress(colored('Epoch {}'.format(epoch), attrs=['reverse']), batches):
            (inputs, input_lens), (targets, target_lens) = batch

            use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO
            loss = model.loss(inputs, input_lens, targets, target_lens, use_teacher_forcing)
            
            L.append(loss.data[0])
            losses.append(loss.data[0])
            if len(losses) > 100:
                loss_text = colored('{:<5.6f}'.format(np.mean(losses[-100:])), 'yellow', attrs=['bold'])
                tqdm.tqdm.write('\033[FLoss: {}'.format(loss_text))
            
            model.zero_grad()
            loss.mean().backward()

            if CLIP_GRAD:
                torch.nn.utils.clip_grad_norm(model.parameters(), CLIP_GRAD)

            optimizer.step()
        
        if epoch in SAVE_POINTS:
            save(save_path.replace('.ckpt', '.epoch{}.ckpt'.format(epoch)))

        epoch += 1
            
    except KeyboardInterrupt:
        done = True
        model.eval()
        test(*batches[0][0])

    save(save_path)

    print('Average loss:', colored(sum(L) / len(L), 'yellow', attrs=['bold']))
    print('Time elapsed:', colored('{}s'.format(int(time.time() - start_time)), 'yellow', attrs=['bold']))


    if done:
        break
