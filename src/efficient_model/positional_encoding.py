import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, max_length: int, dim: int):
        super().__init__()
        
        sin = torch.sin(torch.arange(max_length).unsqueeze(dim=1).float() / (2 * torch.arange((dim + 1) // 2).unsqueeze(dim=0).float() * math.log(10000) / dim).exp())
        cos = torch.cos(torch.arange(max_length).unsqueeze(dim=1).float() / (2 * torch.arange(dim // 2).unsqueeze(dim=0).float() * math.log(10000) / dim).exp())
        
        pe = torch.zeros(size=(max_length, dim))
        pe[:, ::2] = sin
        pe[:, 1::2] = cos
        self.register_buffer('pe', pe, persistent=False)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, cu_seqlen_q: torch.Tensor):
        
        lengths = cu_seqlen_q[1:] - cu_seqlen_q[:-1]
        mask = torch.arange(lengths.max(), device=lengths.device) < lengths.unsqueeze(dim=1)
        indices = torch.nonzero(mask)[:, 1]
        
        return x + self.gate * self.pe[indices]