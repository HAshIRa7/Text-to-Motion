import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import collect_data
import bisect


class HumanoidDataset(Dataset):
    def __init__(self, motions_folder: str, motions_len: int = 500):
        self.data_dict = collect_data(motions_folder)
        self.idx2motion = {}
        for idx, motion_file in enumerate(self.data_dict):
            self.idx2motion[idx] = motion_file
        
        self.motions_len = motions_len
        self.motions_len_cumsum = []
        
        self.mean_velocity: torch.Tensor = torch.zeros(2)
        self.mean_velocity_squared: torch.Tensor = torch.zeros(2)
        self.std_velocity: torch.Tensor | None = None
        
        total_len = 0
        pure_motions_len = 0
        for motion_file in self.data_dict:
            motion_len = self.data_dict[motion_file]['lin_vel'].shape[0]
            pure_motions_len += motion_len
            total_len += (motion_len - self.motions_len)
            self.motions_len_cumsum.append(total_len)
            self.mean_velocity = self.mean_velocity + torch.from_numpy(np.sum(self.data_dict[motion_file]['lin_vel'], axis=0))
            self.mean_velocity_squared = self.mean_velocity_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['lin_vel']**2, axis=0))
            # motion_mean.append(np.mean(self.data_dict[motion_file]['lin_vel']))
            # motion_std.append(np.std(self.data_dict[motion_file]['lin_vel']))
        self.total_len = total_len
        self.mean_velocity = self.mean_velocity / pure_motions_len
        self.mean_velocity_squared = self.mean_velocity_squared / pure_motions_len
        self.std_velocity = torch.sqrt(self.mean_velocity_squared - self.mean_velocity**2)
    
    def __len__(self,):
        return self.total_len - self.motions_len
    
    def __getitem__(self, idx):
        motion_index = bisect.bisect_right(self.motions_len_cumsum, idx)
        if motion_index == 0:
            index_in_motion = idx
        else:
            index_in_motion = idx - self.motions_len_cumsum[motion_index - 1]
        
        motion = self.data_dict[self.idx2motion[motion_index]]
        joint_pos = motion['joint_pos'][index_in_motion:index_in_motion + self.motions_len]
        roll = motion['roll'][index_in_motion:index_in_motion + self.motions_len]
        pitch = motion['pitch'][index_in_motion:index_in_motion + self.motions_len]
        lin_vel = motion['lin_vel'][index_in_motion:index_in_motion + self.motions_len]
        ang_vel = motion['ang_vel'][index_in_motion:index_in_motion + self.motions_len]
        
        return torch.cat([
            torch.tensor(joint_pos).to(dtype=torch.float32),
            torch.tensor(roll[:, None]).to(dtype=torch.float32),
            torch.tensor(pitch[:, None]).to(dtype=torch.float32),
            ((torch.tensor(lin_vel) - self.mean_velocity[None, :]) / self.std_velocity[None, :]).to(dtype=torch.float32),
            torch.tensor(ang_vel[:, None]).to(dtype=torch.float32),
        ], dim=-1)