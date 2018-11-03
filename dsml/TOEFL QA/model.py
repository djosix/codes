import torch
from torch import nn
from torch.nn import functional as F

from module import BiEncoder, Attention


class AttnReader(nn.Module):
    def __init__(self, embed_dim, **params):
        '''
        Args:
        - `embed_dim`: Word embedding dimension.
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.n_hops = params.get('n_hops', 3)
        self.qa_encoder = \
            BiEncoder(embed_dim,
                      dropout=params.get('qa_encoder_dropout', 0))
        self.story_encoder = \
            BiEncoder(embed_dim,
                      dropout=params.get('story_encoder_dropout', 0))
        encode_dim = embed_dim * 2 # (using bidirectional encoder)
        self.attn = \
            Attention(encode_dim, encode_dim,
                      dropout=params.get('attn_dropout', 0))

    def forward(self, story, query, options, output_probs=True):
        '''
        Args: !embedded!
        - `story`: (variable[n_steps, embed_dim], lens).
        - `query`: (variable[n_steps, embed_dim], lens).
        - `options`: A list of (variable[n_steps, embed_dim], lens).

        Returns:
        - `options_prob`: [n_options]. (if output_probs)
        - `options_score`: [n_options]. (if not output_probs)
        '''
        # n_samples = story.size(1)
        # n_options = len(options)
        story, story_len = story
        query, query_len = query
        options, options_len = options
        #============================================================
        # Encoding
        query_feat = self.qa_encoder(query, query_len)
        # [n_samples, encode_dim]
        options_feat = [
            self.qa_encoder(option_embed, seqlen)
            # [n_samples, encode_dim]
            for option_embed, seqlen in zip(options, options_len)
        ]
        story_encoder_outputs, _ = self.story_encoder(story,
                                                      story_len,
                                                      with_outputs=True)
        # [n_steps, n_samples, encode_dim]
        outputs_temp = story_encoder_outputs.permute(1, 0, 2)
        # [n_samples, n_steps, encode_dim]
        #============================================================
        # Hopping & Attention
        context = query_feat
        for _ in range(self.n_hops):
            weights = self.attn(context, story_encoder_outputs)
            # [n_samples, n_steps]
            weights = weights.unsqueeze(-1)
            # [n_samples, n_steps, 1]
            weighted_story_feat = (outputs_temp * weights).sum(1)
            # [n_samples, encode_dim]
            context = context + weighted_story_feat
        #============================================================
        # Probability
        options_score = [
            (option_feat * context).sum(-1).unsqueeze(-1)
            # [n_samples, 1]
            for option_feat in options_feat
        ]
        # pylint: disable=E1101
        options_score = torch.cat(options_score, dim=-1)
        # pylint: enable=E1101
        # [n_samples, n_options]
        if not output_probs:
            return options_score
        options_prob = F.softmax(options_score, dim=-1)
        # [n_samples, n_options]
        return options_prob
