import torch
import torch.nn as nn
import numpy as np
from text_to_motion import FlowMatchingNet, HumanoidDataset
from torch.utils.data import DataLoader
import os
import bisect
from tqdm import tqdm

device = 'cuda:0'
humanoid_dataloader = DataLoader(HumanoidDataset(motions_folder='motions', motions_len = 500), batch_size=64, shuffle=True, drop_last=True)
batch = next(iter(humanoid_dataloader))
first_batch = batch.clone()
flow_net = FlowMatchingNet(input_dim=batch.shape[-1] + 1, hidden_dim=256, output_dim=batch.shape[-1]).to(device)
optimizer = torch.optim.AdamW(flow_net.parameters(), lr=3e-4)
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
        optimizer.step()
        
        loss_sum += loss.item()
        
    print(f'epoch: {epoch}, current_loss: {np.round(loss_sum / len(humanoid_dataloader), 4)}')
        
    if epoch % 10 == 0:
        os.makedirs(save_folder, exist_ok=True)
        torch.save(flow_net.state_dict(), f'{save_folder}/model_weight_{epoch}.pth')
        
        