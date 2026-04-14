import torch
import torch.nn as nn
import numpy as np
from text_to_motion import FlowMatchingNet, HumanoidDataset, TransformerConfig
from torch.utils.data import DataLoader
import os
import bisect
from tqdm import tqdm

device = 'cuda:0'
humanoid_dataset = HumanoidDataset(motions_folder='motions', motions_len = 350, vel_normalization=True)
humanoid_dataloader = DataLoader(humanoid_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=4)
batch = next(iter(humanoid_dataloader))
config = TransformerConfig()
print(f'lin_vel_dataset_mean: {humanoid_dataset.mean_velocity}, lin_vel_dataset_std: {humanoid_dataset.std_velocity}')
flow_net = FlowMatchingNet(
    config=config,
    lin_vel_mean=humanoid_dataset.mean_velocity,
    lin_vel_std=humanoid_dataset.std_velocity,
).to(device)
optimizer = torch.optim.Adam(flow_net.parameters(), lr=2e-5)
save_folder = 'checkpoints'

for epoch in tqdm(range(1000)):
    loss_sum = 0
    for idx, batch in enumerate(humanoid_dataloader):
        x_1 = batch.to(device)
        x_0 = torch.randn_like(x_1)
        
        t = torch.rand(batch.shape[0], 1, 1).to(device=device)
        x_t = t * x_1 + (1 - t) * x_0
        
        optimizer.zero_grad()
        loss = ((flow_net(x_t, t) - (x_1 - x_0))**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=0.5)
        optimizer.step()
        
        loss_sum += loss.item()
        
    print(f'epoch: {epoch}, current_loss: {np.round(loss_sum / len(humanoid_dataloader), 4)}')
        
    if epoch % 5 == 0:
        os.makedirs(save_folder, exist_ok=True)
        torch.save(flow_net.state_dict(), f'{save_folder}/model_weight_{epoch}.pth')
        
        