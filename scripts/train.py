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
from torch.profiler import profile, ProfilerActivity, record_function, schedule

device = 'cuda:0'
dtype=torch.float32
humanoid_dataset = HumanoidDataset(motions_folder='motions')
null_token_embedding = humanoid_dataset.null_token_embedding
humanoid_dataloader = DataLoader(
    humanoid_dataset, 
    batch_size=512, 
    collate_fn=make_collate_fn(null_token_embedding), 
    shuffle=True, 
    drop_last=False, 
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,      
    persistent_workers=True,
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
optimizer = torch.optim.AdamW(flow_net.parameters(), lr=1e-4)
optimizer.zero_grad()
save_folder = 'checkpoints'
scaler = torch.amp.GradScaler('cuda', growth_interval=30000)
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
for epoch in tqdm(range(1000)):
    loss_sum = 0
    my_schedule = schedule(wait=5, warmup=1, active=12)
    with profile(activities=activities, schedule=my_schedule) as profilero:
        pbar = tqdm(enumerate(humanoid_dataloader), total=len(humanoid_dataloader))
        for idx, batch in pbar:
            x_1, cond, cu_seqlen = batch
            x_1 = x_1.to(device=device, dtype=dtype, non_blocking=True)
            cond = cond.to(device=device, dtype=dtype, non_blocking=True)
            cu_seqlen = cu_seqlen.to(device=device, non_blocking=True)
            x_0 = torch.randn_like(x_1)
            
            t = torch.rand(x_1.shape[0], 1).to(dtype=dtype, device=device)
            x_t = t * x_1 + (1 - t) * x_0
            
            # (output_dim == input_dim)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                u_pred = flow_net(x_t, cond, t, cu_seqlen) # (total_q_len, output_dim)
                loss = torch.mean((u_pred - (x_1 - x_0))**2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_sum += loss.detach()
            profilero.step()
            
    print(f'epoch: {epoch}, current_loss: {np.round(loss_sum.item() / len(humanoid_dataset), 4)}')
    loss_sum = 0
    os.makedirs(save_folder, exist_ok=True)
    torch.save(flow_net.state_dict(), f'{save_folder}/model_new_weight_{epoch}.pth')
        
    profilero.export_chrome_trace('trace_gt_0.json')
        