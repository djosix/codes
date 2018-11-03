import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout=0.1):
        """
        Arguments:
        - `hidden_size` (int): Hidden layer dimension.
        - `n_layers` (int): Number of RNN layers.
        - `dropout` (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)


    def forward(self, inputs, input_lengths, with_outputs=False):
        """
        inputs -> gru -> (outputs, hidden)

        Arguments:
        - `inputs` (Variable): Batch of sequences, sorted by sequence lengths,
            [n_steps, batch_size, hidden_size].
        - `input_lengths` (list of int): Sequence lengths.

        Returns:
        - `outputs` (Variable): Bidirectional output sum.
            [n_steps, batch_size, hidden_size]
        - `hidden` (Variable): Last hidden outputs.
            [n_layers, batch_size, hidden_size]
        """
        packed = pack_padded_sequence(inputs, input_lengths)
        outputs, hidden = self.gru(packed, None)
        hidden = hidden.view(self.n_layers, 2, *hidden.size()[-2:]).mean(1)
        if with_outputs:
            outputs, output_lengths = pad_packed_sequence(outputs)
            forward_outputs = outputs[:, :, :self.hidden_size]
            reverse_outputs = outputs[:, :, self.hidden_size:]
            outputs = (forward_outputs + reverse_outputs) / 2
            return outputs, hidden
        else:
            return hidden



class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout=0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, last_hidden):
        # For one time step
        # inputs: [1, batch_size, hidden_size]
        # last_hidden: [n_layers, batch_size, hidden_size]
        assert inputs.size(0) == 1, 'Input should be only a time step'
        gru_output, hidden = self.gru(inputs, last_hidden)
        # gru_output: [1, batch_size, hidden_size]
        # hidden: [n_layers, batch_size, hidden_size]
        gru_output = gru_output.squeeze(0)
        # gru_output: [batch_size, hidden_size]
        output = F.softmax(self.out(gru_output), 1)
        # output: [batch_size, output_size]
        return output, hidden



class Attention(nn.Module):
    """General Luong Attention."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
    

    def forward(self, hidden, encoder_outputs):
        """
        (encoder_outputs -> attn) dot (hidden) -> softmax -> attn_weights

        Arguments:
        - `hidden`: [n_layers * n_directions, batch_size, hidden_size]
        - `encoder_outputs`: [max_len, batch_size, hidden_size]

        Returns:
        - `attn_weights`: Softmaxed attention weights. [batch_size, max_len]
        """
        batch_size = encoder_outputs.size(1)
        max_len = encoder_outputs.size(0)

        attn_outputs = self.attn(encoder_outputs)
        # attn_outputs: [max_len, batch_size, hidden_size]

        # Sum over layers and directions
        hidden_sum = hidden.sum(0, keepdim=True)
        # hidden_sum: [1, batch_size, hidden_size]

        # Dot product on `attn_outputs` and `hidden_sum`
        attn_energies = (attn_outputs * hidden_sum).sum(-1).t()
        # attn_energies: [batch_size, max_len]

        attn_weights = F.softmax(attn_energies, dim=1)
        # attn_weights: [batch_size, max_len]

        return attn_weights



class AttnDecoder(nn.Module):
    """General Luong Attention GRU Decoder."""

    def __init__(self, hidden_size, output_size, n_layers, dropout=0.1):
        """
        Arguments:
        - `hidden_size` (int): RNN hidden layer size.
        - `output_size` (int): The dictionary size.
        - `n_layers`: The number of GRU layers.
        """
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.attn = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, inputs, last_hidden, encoder_outputs):
        """Run one step of RNN decoder.

        Arguments:
        - `inputs`: A batch of index sequences in one time step.
            [1, batch_size, hidden_size]
        - `last_hidden`: Last GRU hidden outputs.
            [n_layers, batch_size, hidden_size]
        - `encoder_outputs`: Encoder output sequences for calculating the
            attention weights. [max_len, batch_size, hidden_size]

        Returns:
        - `output`: [batch_size, output_size]
        - `hidden`: [n_layers, batch_size, hidden_size]
        """
        assert inputs.size(0) == 1, 'Input should be only a time step'
        dropped_inputs = self.dropout(inputs)
        # dropped_inputs: [1, batch_size, hidden_size]

        gru_output, hidden = self.gru(dropped_inputs, last_hidden)
        # gru_output: [1, batch_size, hidden_size]
        # hidden: [n_layers, batch_size, hidden_size]

        attn_weights = self.attn(gru_output, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        # atth_weights: [batch_size, 1, max_len]

        # Apply attention weights to encoder output sequences
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context: [batch_size, 1, hidden_size]

        gru_output = gru_output.squeeze(0)
        # gru_output: [batch_size, hidden_size]

        context = context.squeeze(1)
        # context: [batch_size, hidden_size]

        concat_input = torch.cat([gru_output, context], 1)
        # concat_input: [batch_size, hidden_size * 2]

        concat_output = F.tanh(self.concat(concat_input))
        # concat_output: [batch_size, hidden_size]

        output = F.softmax(self.out(concat_output), 1)
        # output: [batch_size, output_size]

        return output, hidden

