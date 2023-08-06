# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


def build_relative_position(query_size, key_size):
    """
    https://github.com/microsoft/DeBERTa/blob/278392e5a82564485feb3010e215eb79e8ca5bcb/DeBERTa/deberta/disentangled_attention.py#L21
    """

    q_ids = torch.arange(query_size, dtype=torch.long)
    k_ids = torch.arange(key_size, dtype=torch.long)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids.long()

def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = torch.sign(relative_position)
    num_buckets //= 2
    n = torch.abs(relative_position)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret = torch.where(is_small, n, val_if_large) * sign
    return ret


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        rel_pos: bool = True,
        rel_pos_unique: bool = False,
        rp_bucket: int=64,
        max_rp: int=128,
        ur_attn: bool = True,
        ur_attn_unique: bool = False,
        ur_rp_bucket: int=64,
        ur_max_rp: int=128,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        self.num_attention_heads = num_attention_heads
        self.rel_pos = rel_pos
        self.rel_pos_unique = rel_pos_unique
        self.rp_bucket = rp_bucket
        self.max_rp = max_rp
        if self.rel_pos:
            context_position = torch.arange(self.max_seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(self.max_seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position

            if rel_pos_unique:
                self.bucket = relative_position - relative_position.min()
            else:
                self.bucket = relative_position_bucket(
                    relative_position,
                    num_buckets=self.rp_bucket,
                    max_distance=self.max_rp,
                )
                
                self.bucket[self.bucket >= 0] += 2
                self.bucket -= self.bucket.min()

                assert self.bucket.max() <= self.rp_bucket

            self.rp_bias = nn.Embedding(self.rp_bucket + 1, num_attention_heads) if not rel_pos_unique else \
                nn.Embedding(self.bucket.max() + 1, num_attention_heads)

        self.ur_attn = ur_attn
        self.ur_attn_unique = ur_attn_unique
        self.ur_rp_bucket = ur_rp_bucket
        self.ur_max_rp = ur_max_rp
        if self.ur_attn:
            context_position = torch.arange(self.max_seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(self.max_seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position

            if ur_attn_unique:
                self.ur_bucket = relative_position - relative_position.min()
            else:
                self.ur_bucket = relative_position_bucket(
                    relative_position,
                    num_buckets=self.ur_rp_bucket,
                    max_distance=self.ur_max_rp,
                )
                
                self.ur_bucket[self.ur_bucket >= 0] += 2
                self.ur_bucket -= self.ur_bucket.min()

                assert self.ur_bucket.max() <= self.ur_rp_bucket

            self.ur_rp_bias = nn.Embedding(self.ur_rp_bucket + 1, num_attention_heads) if not ur_attn_unique else \
                nn.Embedding(self.ur_bucket.max() + 1, num_attention_heads)

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
    ):
        return TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        rel_pos_bias = None
        if self.rel_pos:
            if self.bucket.device != x.device:
                self.bucket = self.bucket.to(x.device)

            seq_len = x.size(1)
            bucket = self.bucket[:seq_len, :seq_len].contiguous()
            rel_pos_bias = F.embedding(bucket, self.rp_bias.weight)
            rel_pos_bias = rel_pos_bias.permute([2, 0, 1]).contiguous().unsqueeze(0)
            rel_pos_bias = rel_pos_bias.view(1, self.num_attention_heads, seq_len, seq_len).expand(x.size(0), -1, -1, -1).contiguous()
            if padding_mask is not None:
                rel_pos_bias = rel_pos_bias.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            rel_pos_bias = rel_pos_bias.view(-1, seq_len, seq_len)

        ur_rel_pos_bias = None
        if self.ur_attn:
            if self.ur_bucket.device != x.device:
                self.ur_bucket = self.ur_bucket.to(x.device)

            seq_len = x.size(1)
            ur_bucket = self.ur_bucket[:seq_len, :seq_len].contiguous()
            ur_rel_pos_bias = F.embedding(ur_bucket, self.ur_rp_bias.weight)
            ur_rel_pos_bias = ur_rel_pos_bias.permute([2, 0, 1]).contiguous().unsqueeze(0)
            ur_rel_pos_bias = ur_rel_pos_bias.view(1, self.num_attention_heads, seq_len, seq_len).expand(x.size(0), -1, -1,
                                                                                                   -1).contiguous()
            if padding_mask is not None:
                ur_rel_pos_bias = ur_rel_pos_bias.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    0.0,
                )
            ur_rel_pos_bias = ur_rel_pos_bias.view(-1, seq_len, seq_len)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, rel_pos_bias=rel_pos_bias, ur_rel_pos_bias=ur_rel_pos_bias)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
