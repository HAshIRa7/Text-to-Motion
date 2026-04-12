import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        _, seq_len, hidden_dim = x.shape
        
        pe = torch.zeros(seq_len, hidden_dim, device=x.device, dtype=x.dtype)[None, :, :]
        positions = torch.arange(seq_len).to(dtype=x.dtype, device=x.device)
        dimensions = torch.arange(hidden_dim // 2).to(dtype=x.dtype, device=x.device)
        pe[:, :, ::2] = torch.sin(positions[:, None] / (2 * dimensions[None, :] * math.log(10000) / hidden_dim).exp())
        pe[:, :, 1::2] = torch.cos(positions[:, None] / (2 * dimensions[None, :] * math.log(10000) / hidden_dim).exp())
        
        return x + pe

class FlowMatchingNet(nn.Module):
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        lin_vel_mean: torch.Tensor | None = None, 
        lin_vel_std: torch.Tensor | None = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        lin_vel_mean = lin_vel_mean if lin_vel_mean is not None else torch.zeros(2)
        lin_vel_std = lin_vel_std if lin_vel_std is not None else torch.ones(2)
        
        self.register_buffer("lin_vel_mean", lin_vel_mean)
        self.register_buffer("lin_vel_std", lin_vel_std)
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        for i in range(3):
            setattr(self, f'block_{i}', nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True))
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.positional_encoding = PositionEncoding()
        
    def forward(self, x: torch.Tensor, t: torch.tensor):
        '''
        x - size batch_size x seq_len x (input_dim - 1)
        t - size batch_size x 1 x 1
        '''
        input_tensor = torch.cat((x, t.expand(x.shape[0], x.shape[1], 1)), dim=-1)
        x = self.linear1(input_tensor)
        x = self.positional_encoding(x)
        for i in range(3):
            x = getattr(self, f'block_{i}')(x)
        out = self.linear2(x)
        return out
    
    def step(self, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        x_next = x + self.forward(x, t_start[:, None, None]) * (t_end - t_start)[:, None, None]
        return x_next
        