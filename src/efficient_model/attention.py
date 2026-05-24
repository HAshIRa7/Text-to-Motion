"""
Attention with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_to_motion.config import TransformerConfig
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.flash_attn_interface import flash_attn_varlen_func

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
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, cu_seqlen: torch.Tensor, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positionsal embedding to q and k.
        
        Args:
            q: (total_q_len, head_dim)
            k: (total_q_len, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        
        cos = self.cos[:max_length, :]
        sin = self.sin[:max_length, :]

        q_rotated = apply_rotary_emb(q, cos, sin, cu_seqlens=cu_seqlen, max_seqlen=max_length) 
        k_rotated = apply_rotary_emb(k, cos, sin, cu_seqlens=cu_seqlen, max_seqlen=max_length)

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
        cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        total_q_len, H = x.shape
        
        q, k, v = self.qkv_proj(x).split(self.config.hidden_dim, dim=-1)
        q = q.view(total_q_len, self.num_heads, self.head_dim)
        k = k.view(total_q_len, self.num_heads, self.head_dim)
        v = v.view(total_q_len, self.num_heads, self.head_dim)
        
        max_length = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1]).item()
        q, k = self.rope(q, k, cu_seqlen, max_length)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlen,
            cu_seqlens_k=cu_seqlen,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
            dropout_p=self.config.dropout,
            causal=False,
            deterministic=True
        )
        out = out.contiguous().view(total_q_len, H)
        out = self.out_proj(out)

        return out
