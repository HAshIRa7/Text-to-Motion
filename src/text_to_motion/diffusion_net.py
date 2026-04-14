import torch
import torch.nn as nn
import math
from .config import TransformerConfig
from efficient_model import EfficientTransformer

class FlowMatchingNet(nn.Module):
    
    def __init__(
        self,
        config: TransformerConfig,
        lin_vel_mean: torch.Tensor | None = None, 
        lin_vel_std: torch.Tensor | None = None
    ):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        lin_vel_mean = lin_vel_mean if lin_vel_mean is not None else torch.zeros(2)
        lin_vel_std = lin_vel_std if lin_vel_std is not None else torch.ones(2)
        
        self.register_buffer("lin_vel_mean", lin_vel_mean)
        self.register_buffer("lin_vel_std", lin_vel_std)
        
        self.flow_net = EfficientTransformer(config)
        
    def forward(self, x: torch.Tensor, t: torch.tensor):
        '''
        x - size batch_size x seq_len x (input_dim - 1)
        t - size batch_size x 1 x 1
        '''
        return self.flow_net(x, t[:, 0, 0])
    
    def step(self, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        x_next = x + self.forward(x, t_start[:, None, None]) * (t_end - t_start)[:, None, None]
        return x_next
    
    def midpoint_step(self, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor):
        midpoint_vel = self.forward(x + self.forward(x, t_start[:, None, None]) * ((t_end - t_start)[:, None, None]) / 2, (t_start + (t_end - t_start) / 2)[:, None, None])
        x_next = x + (t_end - t_start)[:, None, None] * midpoint_vel
        return x_next
        