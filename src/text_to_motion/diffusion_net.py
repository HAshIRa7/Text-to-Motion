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
        lin_vel_std: torch.Tensor | None = None,
        joint_pos_mean: torch.Tensor | None = None,
        joint_pos_std: torch.Tensor | None = None,
        ang_vel_mean: torch.Tensor | None = None,
        ang_vel_std: torch.Tensor | None = None,
        roll_mean: torch.Tensor | None = None,
        roll_std: torch.Tensor | None = None,
        pitch_mean: torch.Tensor | None = None,
        pitch_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        lin_vel_mean = lin_vel_mean if lin_vel_mean is not None else torch.zeros(2)
        lin_vel_std = lin_vel_std if lin_vel_std is not None else torch.ones(2)
        joint_pos_mean = joint_pos_mean if joint_pos_mean is not None else torch.zeros(29)
        joint_pos_std = joint_pos_std if joint_pos_std is not None else torch.ones(29)
        ang_vel_mean = ang_vel_mean if ang_vel_mean is not None else torch.zeros(1)
        ang_vel_std = ang_vel_std if ang_vel_std is not None else torch.ones(1)
        roll_mean = roll_mean if roll_mean is not None else torch.zeros(1)
        roll_std = roll_std if roll_std is not None else torch.ones(1)
        pitch_mean = pitch_mean if pitch_mean is not None else torch.zeros(1)
        pitch_std = pitch_std if pitch_std is not None else torch.ones(1)
        
        self.register_buffer("lin_vel_mean", lin_vel_mean)
        self.register_buffer("lin_vel_std", lin_vel_std)
        self.register_buffer("joint_pos_mean", joint_pos_mean)
        self.register_buffer("joint_pos_std", joint_pos_std)
        self.register_buffer("ang_vel_mean", ang_vel_mean)
        self.register_buffer("ang_vel_std", ang_vel_std)
        self.register_buffer("roll_mean", roll_mean)
        self.register_buffer("roll_std", roll_std)
        self.register_buffer("pitch_mean", pitch_mean)
        self.register_buffer("pitch_std", pitch_std)
        
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
        