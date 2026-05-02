"""
Attention with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_to_motion.config import TransformerConfig
from flash_attn.layers.rotary import apply_rotary_emb, apply_rotary_emb_torch
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)
        
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache up to seq_len."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)

        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)
    
    def _apply(self, fn):
        super(RotaryPositionalEmbedding, self)._apply(fn)
        self.cos = self.cos.to(device=self.inv_freq.device)
        self.sin = self.sin.to(device=self.inv_freq.device)
        return self
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positionsal embedding to q and k.
        
        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        assert seq_len <= self.max_seq_len, \
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        q_rotated = apply_rotary_emb(q, cos, sin) 
        k_rotated = apply_rotary_emb(k, cos, sin)

        return q_rotated, k_rotated

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with vanilla implementation and RoPE.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout) 
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H = x.shape
        
        q, k, v = self.qkv_proj(x).split(self.config.hidden_dim, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        q, k = self.rope(q, k, S)
        out = flash_attn_func(q, k, v, dropout_p=self.config.dropout, causal=False, deterministic=True)

        out = out.contiguous().view(B, S, H)
        out = self.out_proj(out)

        return out
