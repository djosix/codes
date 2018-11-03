import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import config

from .data import BOS_INDEX
from .models import *


#==================================================
# Utils

def create_mask(seq_lens):
    max_len = max(seq_lens)
    batch_size = len(seq_lens)
    mask = torch.zeros(max_len, batch_size).byte()
    for i in range(batch_size):
        mask[:seq_lens[i], i] = 1
    mask = Variable(mask) # [max_len, batch_size]
    return mask.cuda() if config.CUDA else mask


#==================================================
# Sequence to Sequence Model

class Seq2Seq(nn.Module):
    def __init__(self, n_words=config.DICT_SIZE, hidden_size=512, gru_layers=1, dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.n_words = n_words
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_words, hidden_size)
        self.encoder = Encoder(hidden_size, gru_layers, dropout)
        self.decoder = Decoder(hidden_size, n_words, gru_layers, dropout)

    
    def forward(self, inputs, input_lens, ret_index=True, n_steps=config.MAX_DECODE_LEN):
        # inputs: [n_steps, batch_size]
        # input_lens: [batch_size]
        inputs = inputs.cuda() if config.CUDA else inputs
        encoder_hidden = self.encode(inputs, input_lens)
        outputs = self.decode(encoder_hidden, ret_index=ret_index, ret_steps=False)
        # outputs: [batch_size, n_steps]
        return outputs


    def loss(self, inputs, input_lens, targets, target_lens,
             use_teacher_forcing=True, n_steps=config.MAX_DECODE_LEN):
        # inputs:       [n_steps, batch_size]
        # input_lens:   [batch_size]
        # targets:      [n_steps, batch_size]
        # target_lens:  [batch_size]
        batch_size = inputs.size(1)
        max_len = min(targets.size(0), n_steps)
        
        if max_len < targets.size(0):
            targets = targets[:max_len]
            target_lens = [min(max_len, l) for l in target_lens]

        mask = create_mask(target_lens)[:max_len].float()

        if config.CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()

        encoder_hidden = self.encode(inputs, input_lens)
        teather_inputs = targets if use_teacher_forcing else None
        decoder_outputs = self.decode(encoder_hidden,
                                      teacher_inputs=None,
                                      n_steps=max_len,
                                      ret_index=False,
                                      ret_steps=True)

        picked_outputs = torch.gather(decoder_outputs, 2, targets.unsqueeze(-1))
        cross_entropy = -torch.log(picked_outputs).squeeze(-1)
        masked_cross_entropy = cross_entropy[:mask.size(0)] * mask
        loss = masked_cross_entropy.mean(0) / mask.sum(0)

        return loss


    def encode(self, inputs, input_lens):
        embedded = self.embedding(inputs)
        encoder_hidden = self.encoder(embedded, input_lens)
        return encoder_hidden # [gru_layers, batch_size, hidden_size]


    def decode(self, decoder_hidden, ret_index=True, ret_steps=True,
               n_steps=config.MAX_DECODE_LEN, teacher_inputs=None):
        # decoder_hidden: [gru_layers, batch_size, hidden_size]
        # teacher_inputs: [n_steps, batch_size]
        assert decoder_hidden.size(0) == self.gru_layers
        assert decoder_hidden.size(2) == self.hidden_size
        batch_size = decoder_hidden.size(1)

        if teacher_inputs is not None:
            assert teacher_inputs.size(0) >= n_steps
            assert teacher_inputs.size(1) == batch_size
            teacher_inputs = self.embedding(teacher_inputs)

        input_index = Variable(torch.LongTensor([[BOS_INDEX] * batch_size]))
        input_index = input_index.cuda() if config.CUDA else input_index
        decoder_input = self.embedding(input_index) # [1, batch_size, hidden_size]
        outputs = []

        for t in range(n_steps):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, output_index = decoder_output.data.max(1)
            output_index = output_index.unsqueeze(0) # [1, batch_size]
            output_index = Variable(output_index)
            output_index = output_index.cuda() if config.CUDA else output_index
            decoder_output = decoder_output.unsqueeze(0) # [1, batch_size, n_words]
            outputs.append(output_index if ret_index else decoder_output)

            if teacher_inputs is None:
                input_index = output_index

            else:
                input_index = teacher_inputs[t].unsqueeze(0)

            decoder_input = self.embedding(input_index)

        outputs = torch.cat(outputs, 0) # [n_steps, batch_size, ?]
        outputs = outputs if ret_steps else outputs.transpose(0, 1)

        return outputs


#==================================================
# General Luong Attention Sequence to Sequence Model

class AttnSeq2Seq(nn.Module):
    def __init__(self, n_words=config.DICT_SIZE, hidden_size=512, gru_layers=1, dropout=0.1):
        super(AttnSeq2Seq, self).__init__()
        self.n_words = n_words
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_words, hidden_size)
        self.encoder = Encoder(hidden_size, gru_layers, dropout)
        self.decoder = AttnDecoder(hidden_size, n_words, gru_layers, dropout)

    
    def forward(self, inputs, input_lens, ret_index=True, ret_steps=False, n_steps=config.MAX_DECODE_LEN):
        # inputs: [n_steps, batch_size]
        # input_lens: [batch_size]
        inputs = inputs.cuda() if config.CUDA else inputs
        encoder_outputs, encoder_hidden = self.encode(inputs, input_lens, with_outputs=True)
        outputs = self.decode(encoder_hidden, encoder_outputs, ret_index=ret_index, ret_steps=ret_steps)
        # outputs: [batch_size, n_steps]
        return outputs


    def loss(self, inputs, input_lens, targets, target_lens,
             use_teacher_forcing=True, n_steps=config.MAX_DECODE_LEN):
        # inputs:       [n_steps, batch_size]
        # input_lens:   [batch_size]
        # targets:      [n_steps, batch_size]
        # target_lens:  [batch_size]
        batch_size = inputs.size(1)
        max_len = min(targets.size(0), n_steps)
        
        if max_len < targets.size(0):
            targets = targets[:max_len]
            target_lens = [min(max_len, l) for l in target_lens]

        mask = create_mask(target_lens)[:max_len].float()

        if config.CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
            mask = mask.cuda()

        encoder_outputs, encoder_hidden = self.encode(inputs, input_lens, True)
        teather_inputs = targets if use_teacher_forcing else None
        decoder_outputs = self.decode(encoder_hidden,
                                      encoder_outputs,
                                      teacher_inputs=None,
                                      n_steps=max_len,
                                      ret_index=False,
                                      ret_steps=True)

        picked_outputs = torch.gather(decoder_outputs, 2, targets.unsqueeze(-1))
        cross_entropy = -torch.log(picked_outputs).squeeze(-1)
        masked_cross_entropy = cross_entropy[:mask.size(0)] * mask
        loss = masked_cross_entropy.sum(0) / mask.sum(0)

        return loss


    def encode(self, inputs, input_lens, with_outputs=False):
        embedded = self.embedding(inputs)
        return self.encoder(embedded, input_lens, with_outputs)


    def decode(self, decoder_hidden, encoder_outputs, n_steps=config.MAX_DECODE_LEN,
                ret_index=True, ret_steps=True, teacher_inputs=None):
        # decoder_hidden: [gru_layers, batch_size, hidden_size]
        # teacher_inputs: [n_steps, batch_size]
        assert decoder_hidden.size(0) == self.gru_layers
        assert decoder_hidden.size(2) == self.hidden_size
        batch_size = decoder_hidden.size(1)

        if teacher_inputs is not None:
            assert teacher_inputs.size(0) >= n_steps
            assert teacher_inputs.size(1) == batch_size
            teacher_inputs = self.embedding(teacher_inputs)

        input_index = Variable(torch.LongTensor([[BOS_INDEX] * batch_size]))
        input_index = input_index.cuda() if config.CUDA else input_index
        decoder_input = self.embedding(input_index) # [1, batch_size, hidden_size]
        outputs = []

        for t in range(n_steps):
            decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            _, output_index = decoder_output.data.max(1)
            output_index = output_index.unsqueeze(0) # [1, batch_size]
            output_index = Variable(output_index)
            output_index = output_index.cuda() if config.CUDA else output_index
            decoder_output = decoder_output.unsqueeze(0) # [1, batch_size, n_words]
            outputs.append(output_index if ret_index else decoder_output)

            if teacher_inputs is None:
                input_index = output_index

            else:
                input_index = teacher_inputs[t].unsqueeze(0)

            decoder_input = self.embedding(input_index)

        outputs = torch.cat(outputs, 0) # [n_steps, batch_size, ?]
        outputs = outputs if ret_steps else outputs.transpose(0, 1)

        return outputs
