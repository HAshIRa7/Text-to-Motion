import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import collect_data
import bisect

from .embedder_model import EmbedderModel


class HumanoidDataset(Dataset):
    def __init__(self, motions_folder: str, motions_len: int = 350, vel_normalization=False):
        self.data_dict = collect_data(motions_folder, motions_len)
        self.idx2motion = {}
        for idx, motion_file in enumerate(self.data_dict):
            self.idx2motion[idx] = motion_file
        
        self.motions_len = motions_len
        self.motions_len_cumsum = []
        
        self.mean_velocity: torch.Tensor = torch.zeros(2)
        self.mean_velocity_squared: torch.Tensor = torch.zeros(2)
        self.std_velocity: torch.Tensor = torch.ones(2)
        self.mean_joint_pos: torch.Tensor = torch.zeros(29)
        self.mean_joint_pos_squared: torch.Tensor = torch.zeros(29)
        self.std_joint_pos: torch.Tensor = torch.ones(29)
        self.mean_joint_vel: torch.Tensor = torch.zeros(29)
        self.mean_joint_vel_squared: torch.Tensor = torch.zeros(29)
        self.std_joint_vel: torch.Tensor = torch.ones(29)
        self.mean_ang_vel: torch.Tensor = torch.zeros(1)
        self.mean_ang_vel_squared: torch.Tensor = torch.zeros(1)
        self.std_ang_vel: torch.Tensor = torch.ones(1)
        self.mean_roll: torch.Tensor = torch.zeros(1)
        self.mean_roll_squared: torch.Tensor = torch.ones(1)
        self.std_roll: torch.Tensor = torch.ones(1)
        self.mean_pitch: torch.Tensor = torch.zeros(1)
        self.mean_pitch_squared: torch.Tensor = torch.zeros(1)
        self.std_pitch: torch.Tensor = torch.ones(1)
        self.mean_height: torch.Tensor = torch.zeros(1)
        self.mean_height_squared: torch.Tensor = torch.zeros(1)
        self.std_height: torch.Tensor = torch.ones(1)
        
        total_len = 0
        pure_motions_len = 0
        for motion_file in self.data_dict:
            motion_len = self.data_dict[motion_file]['lin_vel'].shape[0]
            pure_motions_len += motion_len
            total_len += (motion_len - self.motions_len)
            self.motions_len_cumsum.append(total_len)
            if vel_normalization:
                self.mean_ang_vel = self.mean_ang_vel + torch.from_numpy(np.sum(self.data_dict[motion_file]['ang_vel'], axis=0)[None])
                self.mean_ang_vel_squared = self.mean_ang_vel_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['ang_vel']**2, axis=0)[None])
                self.mean_roll = self.mean_roll + torch.from_numpy(np.sum(self.data_dict[motion_file]['roll'], axis=0)[None])
                self.mean_roll_squared = self.mean_roll_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['roll']**2, axis=0)[None])
                self.mean_pitch = self.mean_pitch + torch.from_numpy(np.sum(self.data_dict[motion_file]['pitch'], axis=0)[None])
                self.mean_pitch_squared = self.mean_pitch_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['pitch']**2, axis=0)[None])
                self.mean_joint_pos = self.mean_joint_pos + torch.from_numpy(np.sum(self.data_dict[motion_file]['joint_pos'], axis=0))
                self.mean_joint_pos_squared = self.mean_joint_pos_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['joint_pos']**2, axis=0))
                self.mean_joint_vel = self.mean_joint_vel + torch.from_numpy(np.sum(self.data_dict[motion_file]['joint_vel'], axis=0))
                self.mean_joint_vel_squared = self.mean_joint_vel_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['joint_vel']**2, axis=0))
                self.mean_velocity = self.mean_velocity + torch.from_numpy(np.sum(self.data_dict[motion_file]['lin_vel'], axis=0))
                self.mean_velocity_squared = self.mean_velocity_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['lin_vel']**2, axis=0))
                self.mean_height = self.mean_height + torch.from_numpy(np.sum(self.data_dict[motion_file]['height'], axis=0)[None])
                self.mean_height_squared = self.mean_height_squared + torch.from_numpy(np.sum(self.data_dict[motion_file]['height']**2, axis=0)[None])
        self.total_len = total_len
        if vel_normalization:
            self.mean_ang_vel = self.mean_ang_vel / pure_motions_len
            self.mean_ang_vel_squared = self.mean_ang_vel_squared / pure_motions_len
            self.mean_roll = self.mean_roll / pure_motions_len
            self.mean_roll_squared = self.mean_roll_squared / pure_motions_len
            self.mean_pitch = self.mean_pitch / pure_motions_len
            self.mean_pitch_squared = self.mean_pitch_squared / pure_motions_len
            self.mean_height = self.mean_height / pure_motions_len
            self.mean_height_squared = self.mean_height_squared / pure_motions_len
            self.mean_velocity = self.mean_velocity / pure_motions_len
            self.mean_velocity_squared = self.mean_velocity_squared / pure_motions_len
            self.std_velocity = torch.sqrt(self.mean_velocity_squared - self.mean_velocity**2)
            self.mean_joint_pos = self.mean_joint_pos / pure_motions_len
            self.mean_joint_pos_squared = self.mean_joint_pos_squared / pure_motions_len
            self.std_joint_pos = torch.sqrt(self.mean_joint_pos_squared - self.mean_joint_pos**2)
            self.mean_joint_vel = self.mean_joint_vel / pure_motions_len
            self.mean_joint_vel_squared = self.mean_joint_vel_squared / pure_motions_len
            self.std_joint_vel = torch.sqrt(self.mean_joint_vel_squared - self.mean_joint_vel**2)
            self.std_ang_vel = torch.sqrt(self.mean_ang_vel_squared - self.mean_ang_vel**2)
            self.std_roll = torch.sqrt(self.mean_roll_squared - self.mean_roll**2)
            self.std_pitch = torch.sqrt(self.mean_pitch_squared - self.mean_pitch**2)
            self.std_height = torch.sqrt(self.mean_height_squared - self.mean_height**2)
    
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
        joint_vel = motion['joint_vel'][index_in_motion:index_in_motion + self.motions_len]
        height = motion['height'][index_in_motion:index_in_motion + self.motions_len]
        
        return (torch.cat([
            ((torch.tensor(joint_pos) - self.mean_joint_pos[None, :]) / self.std_joint_pos[None, :]).to(dtype=torch.float32),
            ((torch.tensor(roll[:, None]) - self.mean_roll[None, :]) / self.std_roll[None, :]).to(dtype=torch.float32),
            ((torch.tensor(pitch[:, None]) - self.mean_pitch[None, :]) / self.std_pitch[None, :]).to(dtype=torch.float32),
            ((torch.tensor(lin_vel) - self.mean_velocity[None, :]) / self.std_velocity[None, :]).to(dtype=torch.float32),
            ((torch.tensor(ang_vel[:, None]) - self.mean_ang_vel[None, :]) / self.std_ang_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(joint_vel) - self.mean_joint_vel[None, :]) / self.std_joint_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(height[:, None]) - self.mean_height[None, :]) / self.std_height[None, :]).to(dtype=torch.float32),
        ], dim=-1), motion['text'])
    
    
def make_collate_fn(embedder: EmbedderModel):
    
    def collate_fn(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        proptio_data, text_data = zip(*batch)
        text_embeddings = embedder.encode_text_batch(list(text_data))
        proprio_tensor_data = torch.stack(list(proptio_data))
        
        return proprio_tensor_data, text_embeddings
        
    return collate_fn
