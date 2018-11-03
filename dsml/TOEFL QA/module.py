import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class BiEncoder(nn.Module):
    def __init__(self, dim, dropout=0):
        '''
        Args:
        - `dim` : Hidden layer dimension.
        - `dropout` : Dropout rate.
        '''
        super().__init__()
        self.dim = dim
        self.gru = nn.GRU(dim, dim,
                          dropout=dropout,
                          bidirectional=True)
    
    def forward(self, inputs, input_lens, with_outputs=False):
        '''
        Args:
        - `inputs` : variable[n_steps, n_samples, dim].
        - `input_lens` : variable[n_samples].
        - `with_outpus`
        Returns:
        - `outputs` : Outputs, [n_steps, n_samples, 2 * dim].
        - `hidden` : Hidden weights, [n_samples, 2 * dim].
        '''
        packed, revidxs = self._pack_seqs(inputs, input_lens)
        outputs, hidden = self.gru(packed, None)
        n_samples = inputs.size(1)
        # hidden[2, n_samples, dim]
        hidden = hidden.transpose(0, 1) # [n_samples, 2, dim]
        hidden = hidden.contiguous()
        # handle non-contiguity caused by transpose()
        hidden = hidden.view(n_samples, 2 * self.dim)
        hidden = hidden[revidxs, :] # recover order
        if with_outputs:
            outputs, _ = pad_packed_sequence(outputs)
            outputs = outputs[:, revidxs, :] # recover order
            return outputs, hidden
        else:
            return hidden
    
    def _pack_seqs(self, seqs, seqlens):
        seqlens, idxs = seqlens.sort(0, True)
        revidxs = idxs.sort()[1] # for recovering order
        seqs = seqs[:, idxs, :] # sort seqs
        packed = pack_padded_sequence(seqs, seqlens.tolist())
        return packed, revidxs


class Attention(nn.Module):
    def __init__(self, feature_dim, encoder_output_dim, dropout=0):
        '''
        Args:
        - `feature_dim` : Dimension of the features used to compute
            the attention with outputs.
        - `encoder_output_dim` : Dimension of the encoder output.
        '''
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder_output_dim = encoder_output_dim
        self.attn = nn.Linear(encoder_output_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, encoder_outputs):
        '''
        Args:
        - `features` : Features for computing attention weights,
            [n_samples, feature_dim].
        - `encoder_outputs` : [n_steps, n_samples, encoder_output_dim].
        Returns:
        - `attn_weights` : Weights of the encoder outputs,
            [n_steps, n_samples].
        '''
        # n_samples = features.size(0)
        # n_steps = encoder_outputs.size(0)
        attn_outputs = self.attn(encoder_outputs)
        attn_outputs = self.dropout(attn_outputs)
        # [n_steps, n_samples, feature_dim]
        features = features.unsqueeze(0)
        # [1, n_samples, feature_dim]
        energies = attn_outputs * features
        # [n_steps, n_samples, feature_dim]
        energies = energies.sum(2).t()
        # [n_samples, n_steps]
        weights = F.softmax(energies, dim=1)
        return weights
