"""
Efficient Transformer Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_to_motion.config import TransformerConfig
from efficient_model.norm import RMSNorm
from efficient_model.swiglu import SwiGLUFeedForward
from efficient_model.attention import MultiHeadAttention, MultiHeadCrossAttention
from efficient_model.adaln import FusedAdaLNModulation
from efficient_model.positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.cross_attn = MultiHeadCrossAttention(config)
        self.ln3 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cu_seqlen_q: torch.Tensor,
        cu_seqlen_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cu_seqlen_q, max_length_q)
        x = x + self.cross_attn(self.ln2(x), cond, cu_seqlen_q, cu_seqlen_k, max_length_q, max_length_k)
        x = x + self.ffn(self.ln3(x))
        return x


class EfficientTransformer(nn.Module):
    """
    Efficient Transformer language model.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.absolute_position_encoding = PositionalEncoding(config.max_seq_len, config.hidden_dim)
        self.in_linear = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.adaln_layer = FusedAdaLNModulation(config.hidden_dim)
        self.out_linear = nn.Linear(config.hidden_dim, config.output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        cu_seqlen_q: torch.Tensor,
        cu_seqlen_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (total_q_len, input_dim) token indices
            cond: (toral_k_len, embed_dim)
            t: (total_q_len,)
            cu_seqlen_q: (batch_size + 1,)
            cu_seqlen_k: (batch_size + 1,)
        Returns:
            pred: (total_q_len, output_dim)
        """
        x = self.in_linear(x)
        x = self.absolute_position_encoding(x, cu_seqlen_q)
        for idx, layer in enumerate(self.layers):
            x = layer(x, cond, cu_seqlen_q, cu_seqlen_k, max_length_q, max_length_k)
            x = self.adaln_layer(x, t)
        x = self.out_linear(x)
        return x
