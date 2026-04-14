import torch
import torch.nn as nn

class TimeStepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freqs = torch.exp(-torch.arange(0, dim, 2) * (4.4 / dim))
        self.register_buffer('freqs', freqs)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        x - batch_size x seq_len x dim
        t - batch_size 
        '''
        t_emb = t.unsqueeze(dim=-1) * self.freqs.unsqueeze(dim=0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        gamma, beta = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(dim=1)
        beta = beta.unsqueeze(dim=1)
        return self.norm(x) * (1 + gamma) + beta