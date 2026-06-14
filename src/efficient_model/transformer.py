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

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

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
        self.adaln_layer = FusedAdaLNModulation(freq_dim=config.hidden_dim, dim=config.hidden_dim)
        self.modulation = nn.Parameter(torch.randn(1, 6 * config.hidden_dim) / config.hidden_dim**0.5)
        
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

        in_dtype = x.dtype
        
        (sh_sa, sc_sa, g_sa, sh_ff, sc_ff, g_ff) = (
            self.adaln_layer(t) + self.modulation
        ).chunk(6, dim=-1)

        attn_in = modulate(self.ln1(x), sh_sa, sc_sa)
        y = self.attn(attn_in, cu_seqlen_q, max_length_q)
        with torch.autocast(device_type="cuda", enabled=False):
            x = (x + g_sa.float() * y.float()).to(in_dtype)

        cross_in = self.ln2(x)
        x = x + self.cross_attn(
            cross_in, cond, cu_seqlen_q, cu_seqlen_k, max_length_q, max_length_k
        )

        ffn_in = modulate(self.ln3(x), sh_ff, sc_ff)
        y = self.ffn(ffn_in)
        with torch.autocast(device_type="cuda", enabled=False):
            x = (x + g_ff.float() * y.float()).to(in_dtype)

        return x


class EfficientTransformer(nn.Module):
    """
    Efficient Transformer language model.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.in_linear = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.last_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.out_linear = nn.Linear(config.hidden_dim, config.output_dim)
        self.apply(self._init_weights)
        for layer in self.layers:
            nn.init.zeros_(layer.adaln_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaln_layer.adaLN_modulation[-1].bias)

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
        for idx, layer in enumerate(self.layers):
            x = layer(x, cond, t, cu_seqlen_q, cu_seqlen_k, max_length_q, max_length_k)
        x = self.out_linear(self.last_norm(x))
        return x
