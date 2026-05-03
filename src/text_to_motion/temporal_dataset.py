import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import collect_data, last_token_pool
import bisect
from typing import Dict
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

g1_data_names2size = {
    'velocity': 2,
    'joint_pos': 29,
    'joint_vel': 29,
    'ang_vel': 1,
    'roll': 1,
    'pitch': 1,
    'height': 1,
}

def Qwen_embed_data(data: Dict, device='cuda:0'):
    '''
    data - dict with mapping file_name to dict with key 'text'
    return - list torch.Tensor, null_token
    '''
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B').to(device)
    model.eval()
    max_length = 8192

    data_len = len(data)
    all_texts = [motion['text'] for motion in data.values()]
    batch_size = 64
    list_embeddings = []
    for it in tqdm(range(data_len // batch_size + 1)):
        batch = all_texts[it * batch_size : min(data_len, it * batch_size + batch_size)]
        if it == data_len // batch_size:
            batch = batch + ['']
        batch_dict = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        del outputs, batch_dict
        if it % 32 == 0:
            torch.cuda.empty_cache()
        list_embeddings.extend([embeddings[i].cpu() for i in range(len(batch))])
    
    del model
    torch.cuda.empty_cache()
    return list_embeddings, list_embeddings[-1]

class HumanoidDataset(Dataset):
    def __init__(self, motions_folder: str, motions_len_min: int = 100, motions_len_max: int = 500, catch_temporal=False):
        self.data_dict = collect_data(motions_folder, motions_len_min, motions_len_max)
        
        self.idx2motion = {}
        for idx, motion_file in enumerate(self.data_dict):
            self.idx2motion[idx] = motion_file
        
        if not catch_temporal:
            self.embeddings, self.null_token_embedding = Qwen_embed_data(self.data_dict)
        else:
            pass
        
        for k, siz in g1_data_names2size.items():
            setattr(self, f'mean_{k}', torch.zeros(siz))
            setattr(self, f'mean_{k}_squared', torch.zeros(siz))
            setattr(self, f'std_{k}', torch.ones(siz))
        pure_motions_len = 0
        for motion_file in self.data_dict:
            pure_motions_len += self.data_dict[motion_file]['velocity'].shape[0]
            for k in g1_data_names2size:
                aggregated_first_momentum = np.sum(self.data_dict[motion_file][k], axis=0)
                aggregated_second_momentum = np.sum(self.data_dict[motion_file][k]**2, axis=0) 
                if not isinstance(aggregated_first_momentum, np.ndarray):
                    aggregated_first_momentum = aggregated_first_momentum[None]
                    aggregated_second_momentum = aggregated_second_momentum[None]
                setattr(self, f'mean_{k}', getattr(self, f'mean_{k}') + torch.from_numpy(aggregated_first_momentum))
                setattr(self, f'mean_{k}_squared', getattr(self, f'mean_{k}_squared') + torch.from_numpy(aggregated_second_momentum))
        for k in g1_data_names2size:
            setattr(self, f'mean_{k}', getattr(self, f'mean_{k}') / pure_motions_len)
            setattr(self, f'mean_{k}_squared', getattr(self, f'mean_{k}_squared') / pure_motions_len)
            setattr(self, f'std_{k}', torch.sqrt(getattr(self, f'mean_{k}_squared') - getattr(self, f'mean_{k}')**2))
    
    def __len__(self,):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        
        motion = self.data_dict[self.idx2motion[idx]]
        
        joint_pos = motion['joint_pos']
        roll = motion['roll']
        pitch = motion['pitch']
        lin_vel = motion['velocity']
        ang_vel = motion['ang_vel']
        joint_vel = motion['joint_vel']
        height = motion['height']

        return (torch.cat([
            ((torch.tensor(joint_pos) - self.mean_joint_pos[None, :]) / self.std_joint_pos[None, :]).to(dtype=torch.float32),
            ((torch.tensor(roll[:, None]) - self.mean_roll[None, :]) / self.std_roll[None, :]).to(dtype=torch.float32),
            ((torch.tensor(pitch[:, None]) - self.mean_pitch[None, :]) / self.std_pitch[None, :]).to(dtype=torch.float32),
            ((torch.tensor(lin_vel) - self.mean_velocity[None, :]) / self.std_velocity[None, :]).to(dtype=torch.float32),
            ((torch.tensor(ang_vel[:, None]) - self.mean_ang_vel[None, :]) / self.std_ang_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(joint_vel) - self.mean_joint_vel[None, :]) / self.std_joint_vel[None, :]).to(dtype=torch.float32),
            ((torch.tensor(height[:, None]) - self.mean_height[None, :]) / self.std_height[None, :]).to(dtype=torch.float32),
        ], dim=-1), self.embeddings[idx])
    
    
def make_collate_fn(null_token_embedding: torch.Tensor, replacement_probs: float = 0.15):
    
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_len = len(batch)
        proprio_list_data, embeddings_list_data = zip(*batch)
        mask = torch.rand(batch_len) > replacement_probs
        cumsum_seq_lens = torch.cumsum(
            torch.tensor([len(proprio) for proprio in proprio_list_data]),
            dim=0,
        )
        cu_seqlen = torch.cat((torch.zeros(1), cumsum_seq_lens), dim=0).to(dtype=torch.int32)
        proprio_tensor_data = torch.cat(proprio_list_data, dim=0) # shape - total_len_q x proprio_dim
        embeddings_tensor_data = torch.stack(embeddings_list_data) # shape - batch_size x embedding_dim
        embeddings_tensor_data = torch.where(mask.unsqueeze(dim=-1), embeddings_tensor_data, null_token_embedding.unsqueeze(dim=0))
        embeddings_tensor_data = torch.repeat_interleave(embeddings_tensor_data, cu_seqlen[1:] - cu_seqlen[:-1], dim=0)
    
        return (
            proprio_tensor_data, 
            embeddings_tensor_data,
            cu_seqlen
        )
        
    return collate_fn
