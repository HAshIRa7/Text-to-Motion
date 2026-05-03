import torch
import torch.nn as nn

class TimeStepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freqs = torch.exp(-torch.arange(0, dim, 2) * (4.4 / dim))
        self.register_buffer('freqs', freqs)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 2 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @torch.compile
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        x - total_q_len x dim
        t - total_q_len
        '''
        t_emb = t.unsqueeze(dim=-1) * self.freqs.unsqueeze(dim=0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        gamma, beta = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta 
    
    
class ConditionEmbedder(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(embed_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        nn.init.zeros_(self.module[-1].weight)
        nn.init.zeros_(self.module[-1].bias)
        
    @torch.compile
    def forward(self, x: torch.Tensor, embeding: torch.Tensor):
        '''
        x - total_q_len x dim
        embedding - total_q_len x hidden_dim
        '''
        gamma, beta = self.module(embeding).chunk(2, dim=-1) # gamma shape - total_q_len x dim, beta shape - total_q_len x dim
        return self.norm(x) * (1 + gamma) + beta