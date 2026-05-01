import torch
import torch.nn as nn
import numpy as np
from text_to_motion import (
    FlowMatchingNet, 
    HumanoidDataset,
    TransformerConfig, 
    make_collate_fn,
)
from torch.utils.data import DataLoader
import os
import bisect
from tqdm import tqdm

device = 'cuda:0'
dtype=torch.bfloat16
humanoid_dataset = HumanoidDataset(motions_folder='motions', motions_len = 350, vel_normalization=True)
null_token_embedding = humanoid_dataset.null_token_embedding
humanoid_dataloader = DataLoader(
    humanoid_dataset, 
    batch_size=256, 
    collate_fn=make_collate_fn(null_token_embedding), 
    shuffle=True, 
    drop_last=False, 
    num_workers=4
)
batch = next(iter(humanoid_dataloader))
config = TransformerConfig(input_dim=batch[0].shape[-1], embed_dim=len(null_token_embedding), output_dim=batch[0].shape[-1])
print(f'lin_vel_dataset_mean: {humanoid_dataset.mean_velocity}, lin_vel_dataset_std: {humanoid_dataset.std_velocity}')
print(f'joint_pos_dataset_mean: {humanoid_dataset.mean_joint_pos}, joint_pos_dataset_std: {humanoid_dataset.std_joint_pos}')
print(f'ang_vel_dataset_mean: {humanoid_dataset.mean_ang_vel}, ang_vel_dataset_std: {humanoid_dataset.std_ang_vel}')
print(f'roll_dataset_mean: {humanoid_dataset.mean_roll}, roll_dataset_std: {humanoid_dataset.std_roll}')
print(f'pitch_dataset_mean: {humanoid_dataset.mean_pitch}, pitch_dataset_std: {humanoid_dataset.std_pitch}')
print(f'height_dataset_mean: {humanoid_dataset.mean_height}, pitch_dataset_std: {humanoid_dataset.std_height}')
flow_net = FlowMatchingNet(
    config=config,
    lin_vel_mean=humanoid_dataset.mean_velocity,
    lin_vel_std=humanoid_dataset.std_velocity,
    joint_pos_mean=humanoid_dataset.mean_joint_pos,
    joint_pos_std=humanoid_dataset.std_joint_pos,
    ang_vel_mean=humanoid_dataset.mean_ang_vel,
    ang_vel_std=humanoid_dataset.std_ang_vel,
    roll_mean=humanoid_dataset.mean_roll,
    roll_std=humanoid_dataset.std_roll,
    pitch_mean=humanoid_dataset.mean_pitch,
    pitch_std=humanoid_dataset.std_pitch,
    joint_vel_mean=humanoid_dataset.mean_joint_vel,
    joint_vel_std=humanoid_dataset.std_joint_vel,
    height_mean=humanoid_dataset.mean_height,
    height_std=humanoid_dataset.std_height,
).to(dtype=dtype, device=device)
optimizer = torch.optim.Adam(flow_net.parameters(), lr=3e-4)
save_folder = 'checkpoints'
for epoch in tqdm(range(1000)):
    loss_sum = 0
    for idx, batch in tqdm(enumerate(humanoid_dataloader)):
        x_1, cond = batch
        x_1 = x_1.to(device=device, dtype=dtype)
        cond = cond.to(device=device, dtype=dtype)
        x_0 = torch.randn_like(x_1)
        
        t = torch.rand(x_1.shape[0], 1, 1).to(dtype=dtype, device=device)
        
        x_t = t * x_1 + (1 - t) * x_0
        
        optimizer.zero_grad()
        u_pred = flow_net(x_t, cond, t)
        loss = torch.mean((u_pred - (x_1 - x_0))**2) 
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        
        if idx % 1000 == 0:
            print(f'epoch: {epoch}, current_loss: {np.round(loss_sum / 1000, 4)}')
            loss_sum = 0
            os.makedirs(save_folder, exist_ok=True)
            torch.save(flow_net.state_dict(), f'{save_folder}/model_weight_{epoch}_{idx}.pth')
        
        