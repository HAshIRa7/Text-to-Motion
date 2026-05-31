import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import collect_data, last_token_pool
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pickle

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
    def __init__(self, motions_folder: str, motions_new_folder: str, motions_len_min: int = 51, motions_len_max: int = 2500, catch_temporal=False):
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
            emb = torch.tensor(motion['emb'])

        return (torch.cat([
            ((torch.tensor(joint_pos) - self.statsCollector.mean_joint_pos[None, :]) / self.statsCollector.std_joint_pos[None, :]).to(dtype=torch.float32),
            ((torch.tensor(roll[:, None]) - self.statsCollector.mean_roll[None, :]) / self.statsCollector.std_roll[None, :]).to(dtype=torch.float32),
            ((torch.tensor(pitch[:, None]) - self.statsCollector.mean_pitch[None, :]) / self.statsCollector.std_pitch[None, :]).to(dtype=torch.float32),
            ((torch.tensor(lin_vel) - self.statsCollector.mean_velocity[None, :]) / self.statsCollector.std_velocity[None, :]).to(dtype=torch.float32),
            ((torch.tensor(ang_vel[:, None]) - self.statsCollector.mean_ang_vel[None, :]) / self.statsCollector.std_ang_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(joint_vel) - self.statsCollector.mean_joint_vel[None, :]) / self.statsCollector.std_joint_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(height[:, None]) - self.statsCollector.mean_height[None, :]) / self.statsCollector.std_height[None, :]).to(dtype=torch.float32),
        ], dim=-1), emb)
    
    
def make_collate_fn(null_token_embedding: torch.Tensor, replacement_probs: float = 0.08):
    
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_list_data, embeddings_list_data = zip(*batch)
        cumsum_seq_lens = torch.cumsum(
            torch.tensor([len(proprio) for proprio in proprio_list_data]),
            dim=0,
        )
        cu_seqlen_q = torch.cat((torch.zeros(1), cumsum_seq_lens), dim=0).to(dtype=torch.int32)
        proprio_tensor_data = torch.cat(proprio_list_data, dim=0) # shape - total_len_q x proprio_dim
        mask = torch.rand(len(embeddings_list_data)) < replacement_probs
        unsq_null_token_embedding = null_token_embedding.unsqueeze(0)
        new_embeddings_list_data = [
            unsq_null_token_embedding if elem_mask else seqs_embeddings
            for elem_mask, seqs_embeddings in zip(mask, embeddings_list_data)
        ]
        text_cumsum_seq_lens = torch.cumsum(
            torch.tensor([len(emb) for emb in new_embeddings_list_data]),
            dim=0,
        )
        cu_seqlen_k = torch.cat((torch.zeros(1), text_cumsum_seq_lens), dim=0).to(dtype=torch.int32)
        embeddings_tensor_data = torch.cat(new_embeddings_list_data, dim=0)
    
        return (
            proprio_tensor_data, 
            embeddings_tensor_data,
            cu_seqlen_q,
            cu_seqlen_k,
            len(batch),
        )
        
    return collate_fn



