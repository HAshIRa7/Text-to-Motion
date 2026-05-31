import torch
import torch.nn as nn
import math


class FusedAdaLNModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        freqs = torch.exp(-torch.arange(0, dim, 2) * (math.log(10000) / (dim // 2 - 1)))
        self.register_buffer('freqs', freqs)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 3 * dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        gamma, beta, alpha = self.adaLN_modulation(t_emb).chunk(3, dim=-1)
        return x + alpha * (self.norm(x) * (1 + gamma) + beta)