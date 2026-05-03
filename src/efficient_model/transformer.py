"""
Efficient Transformer Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_to_motion.config import TransformerConfig
from efficient_model.norm import RMSNorm
from efficient_model.swiglu import SwiGLUFeedForward
from efficient_model.attention import MultiHeadAttention
from efficient_model.adaln import TimeStepEmbedder, ConditionEmbedder


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config.hidden_dim, config.intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cu_seqlen)
        x = x + self.ffn(self.ln2(x))
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
        
        self.adaln_layers = nn.ModuleList([
            TimeStepEmbedder(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        self.text_adaln_layers = nn.ModuleList([
            ConditionEmbedder(config.embed_dim, config.hidden_dim) for _ in range(config.num_layers)
        ])

        # self.ln_f = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
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
        cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (total_q_len, input_dim) token indices
            cond: (B, embed_dim)
            t: (total_q_len,)
            cu_seqlen: (batch_size + 1,)
        Returns:
            pred: (total_q_len, output_dim)
        """
        x = self.in_linear(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x, cu_seqlen)
            x = self.adaln_layers[idx](x, t)
            x = self.text_adaln_layers[idx](x, cond)
        x = self.out_linear(x)
        return x
