import importlib
import itertools
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.autograd import Variable

from rl_chatbot import config, data, models, seq2seq_models, utils
from termcolor import colored


LOAD = 'seq2seq_chinese_next_utils.jieba_tokenizer_0.5_512_1_0.3_10.epoch500'   
SAVE = 'rl_fine_tuning'
CLIP_GRAD = 10
MAX_TIME_STEP = 200

# Weight of reward sum
ALPHA = 0.5

print(LOAD, '=>', SAVE)

#==================================================
# Load

corpus_options = {
    'name': 'chinese',
    'dict_size': config.DICT_SIZE,
    'tokenizer': utils.jieba_tokenizer,
    'target': 'next',
    'file_filter': lambda name: name.endswith('.txt'),
    'use_cache': True
}

model_options = {
    'n_words': config.DICT_SIZE,
    'hidden_size': 512,
    'gru_layers': 1,
    'dropout': 0.1
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
    model.load_state_dict(states['model'])
    # optimizer.load_state_dict(states['optimizer'])

else:
    print('New session')

if config.CUDA:
    model.cuda()



#==================================================
# Functions

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



#==================================================
# Training


def calculate_seq_lens(seqs, plus_eos=True): # step first
    max_len = seqs.size(0)
    seq_lens = torch.LongTensor(*seqs.size()[1:]) * 0
    end = torch.ByteTensor(*seqs.size()[1:]) * 0
    for step in seqs:
        end = end | (step < 4)
        seq_lens += (end ^ 1).long()
    if plus_eos:
        seq_lens += (seq_lens < max_len).long()
    return seq_lens

def sample_from_outputs(outputs, num): # step first
    return torch.cat([torch.multinomial(probs, num).unsqueeze(0)
                      for probs in outputs])


batch_size = 64
done = False
episode = 1

RR = []

for batches in utils.general_batches_generator(corpus, batch_size, 'shuffle'):

    start_time = time.time()

    print('Reward: --')

    try:
        for batch in batches:
            R = []

            model.eval()
            test(*batch[0])
            model.train()

            # episode
            (inputs, input_lens), _ = batch
            inputs = inputs.cuda() if config.CUDA else inputs

            print()

            for time_step in utils.progress('Episode {}'.format(episode), range(MAX_TIME_STEP)):
                outputs = model(inputs, input_lens, ret_index=False, ret_steps=True)
                # [n_steps, batch_size, n_words]
                actions = sample_from_outputs(outputs, 1)
                # [n_steps, batch_size, 1]
                action_lens = calculate_seq_lens(actions.cpu().data.squeeze(-1))
                # [batch_size]
                mask = seq2seq_models.create_mask(action_lens)
                # [n_steps, batch_size]
                probs = torch.gather(outputs, -1, actions).squeeze(-1)[:mask.size(0)]
                # [n_steps, batch_size]
                probs = (probs * mask.float()).mean(0)
                # [batch_size]

                # ITS TOO UGLY DO NOT SEE THIS PLEASE
                repeat_count = torch.zeros(batch_size)
                repeat_count = repeat_count.cuda() if config.CUDA else repeat_count
                actions = actions.squeeze(-1)
                actions_tensor = actions.data
                current = actions_tensor[0]
                for step in actions_tensor[1:]:
                    repeat_count += ((step == current) & (step >= 4)).float()
                    current = step
                action_len_rewards = action_lens.float() 
                action_len_rewards = action_len_rewards.cuda() if config.CUDA else action_len_rewards
                repeat_count_rewards = -repeat_count / action_len_rewards # repeated / length
                action_len_rewards = action_len_rewards / config.MAX_DECODE_LEN # length / max length

                rewards = ALPHA * repeat_count_rewards + (1 - ALPHA) * action_len_rewards
                rewards = rewards * probs.data
                rewards = Variable(rewards)
                rewards = rewards.cuda() if config.CUDA else rewards
                rewards_mean = rewards.mean().data[0]
                rewards = rewards - rewards_mean # baseline
                # [batch_size]
                
                loss = model.loss(inputs, input_lens, actions, action_lens.tolist())

                model.zero_grad()
                (loss * rewards).mean().backward()
                optimizer.step()

                inputs = model(inputs, input_lens, ret_steps=True)
                input_lens = calculate_seq_lens(inputs.cpu().data)
                input_lens, i = input_lens.sort(descending=True)
                inputs = inputs[:, i.cuda() if config.CUDA else i]
                input_lens = input_lens.tolist()
            
                R.append(rewards_mean)

                reward_text = colored('{:<5.6f}'.format(rewards_mean), 'yellow', attrs=['bold'])
                tqdm.tqdm.write('\033[FReward: {}'.format(reward_text))
            
            print(colored('Average reward: {}\n'.format(np.mean(R)), 'green', attrs=['bold']))
            RR += R

            episode += 1

            
            
    except KeyboardInterrupt:
        done = True
        model.eval()
        test(*batches[0][0])
        print('Saving...')


    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)

    if done:
        break
