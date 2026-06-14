import torch
import torch.nn as nn
import math


class FusedAdaLNModulation(nn.Module):
    def __init__(self, freq_dim, dim):
        super().__init__()
        freqs = torch.exp(-torch.arange(0, freq_dim, 2) * (math.log(10000) / (freq_dim // 2 - 1)))
        self.register_buffer('freqs', freqs)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, t: torch.Tensor):
        t_emb = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return  self.adaLN_modulation(t_emb)