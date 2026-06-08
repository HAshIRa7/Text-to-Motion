import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import collect_data, last_token_pool
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pickle
from transformers import AutoTokenizer
import random

g1_data_names2size = {
    'velocity': 2,
    'joint_pos': 29,
    'joint_vel': 29,
    'ang_vel': 1,
    'roll': 1,
    'pitch': 1,
    'height': 1,
}

class HumanoidDataset(Dataset):
    def __init__(self, motions_folder: str, motions_new_folder: str):
        self.main_motions_new_folder = motions_new_folder
        with open('statistic_collector.pkl', 'rb') as statistic_collector:
            self.statsCollector = pickle.load(statistic_collector)
        with np.load('new_null_token_embedding.npz', allow_pickle=True) as embedding:
            self.null_token_embedding = torch.tensor(embedding['embedding'].copy())

        self.motion_names = os.listdir(motions_new_folder) 
        
    def __len__(self,):
        return len(self.motion_names)
    
    def __getitem__(self, idx):
        with np.load(os.path.join(self.main_motions_new_folder, self.motion_names[idx]), allow_pickle=True) as motion:
            joint_pos = motion['joint_pos']
            roll = motion['roll']
            pitch = motion['pitch']
            lin_vel = motion['velocity']
            ang_vel = motion['ang_vel']
            joint_vel = motion['joint_vel']
            height = motion['height']
            text = motion['text'].item()

        return (torch.cat([
            ((torch.tensor(joint_pos) - self.statsCollector.mean_joint_pos[None, :]) / self.statsCollector.std_joint_pos[None, :]).to(dtype=torch.float32),
            ((torch.tensor(roll[:, None]) - self.statsCollector.mean_roll[None, :]) / self.statsCollector.std_roll[None, :]).to(dtype=torch.float32),
            ((torch.tensor(pitch[:, None]) - self.statsCollector.mean_pitch[None, :]) / self.statsCollector.std_pitch[None, :]).to(dtype=torch.float32),
            ((torch.tensor(lin_vel) - self.statsCollector.mean_velocity[None, :]) / self.statsCollector.std_velocity[None, :]).to(dtype=torch.float32),
            ((torch.tensor(ang_vel[:, None]) - self.statsCollector.mean_ang_vel[None, :]) / self.statsCollector.std_ang_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(joint_vel) - self.statsCollector.mean_joint_vel[None, :]) / self.statsCollector.std_joint_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(height[:, None]) - self.statsCollector.mean_height[None, :]) / self.statsCollector.std_height[None, :]).to(dtype=torch.float32),
        ], dim=-1), text)


def make_collate_fn(tokenizer: AutoTokenizer, replacement_probs: float = 0.15):
    
    def collate_fn(batch: list[tuple[torch.Tensor, str]]):
        proprio_list_data, text_list_data = zip(*batch)
        cumsum_seq_lens = torch.cumsum(
            torch.tensor([len(proprio) for proprio in proprio_list_data]),
            dim=0,
        )
        cu_seqlen_q = torch.cat((torch.zeros(1), cumsum_seq_lens), dim=0).to(dtype=torch.int32)
        proprio_tensor_data = torch.cat(proprio_list_data, dim=0) # shape - total_len_q x proprio_dim
        mask = [random.random() > replacement_probs for _ in range(len(text_list_data))]
        new_text_list_data = [text if elem_mask else '' for text, elem_mask in zip(text_list_data, mask)]
        text_batch = tokenizer(new_text_list_data, padding=True, truncation=True, max_length=512, return_tensors="pt")

        seq_lens = text_batch['attention_mask'].sum(dim=1)
        cu_seqlen_k = torch.cat((torch.zeros(1), torch.cumsum(seq_lens, dim=-1)), dim=0).to(dtype=torch.int32)
        max_length_q = max(list(map(len, proprio_list_data)))
        max_length_k = max(seq_lens.tolist())
    
        return (
            proprio_tensor_data,
            text_batch,
            cu_seqlen_q,
            cu_seqlen_k,
            len(batch),
            max_length_q,
            max_length_k,
        )
        
    return collate_fn



