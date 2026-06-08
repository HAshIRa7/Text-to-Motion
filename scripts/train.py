import torch
import torch.nn as nn
import numpy as np
from text_to_motion import (
    FlowMatchingNet, 
    HumanoidDataset,
    TransformerConfig, 
    make_collate_fn,
)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import bisect
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoTokenizer, T5EncoderModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_setup():
    """Initialize the process group. Expects torchrun-provided env vars."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def main():
    
    dtype=torch.float32
    batch_size=16
    local_rank = ddp_setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"


    text_encoder_model_name = 'google/flan-t5-xl'
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_name)
    humanoid_dataset = HumanoidDataset(motions_folder='motions', motions_new_folder='postprocessed_motions')
    # null_token_embedding = humanoid_dataset.null_token_embedding 
    
    sampler = DistributedSampler(
        humanoid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    humanoid_dataloader = DataLoader(
        humanoid_dataset, 
        batch_size=batch_size, 
        collate_fn=make_collate_fn(tokenizer=tokenizer), 
        sampler=sampler,
        drop_last=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    batch = next(iter(humanoid_dataloader))
    config = TransformerConfig(input_dim=batch[0].shape[-1], embed_dim=2048, output_dim=batch[0].shape[-1])
    if is_main_process():
        print(f'lin_vel_dataset_mean: {humanoid_dataset.statsCollector.mean_velocity}, lin_vel_dataset_std: {humanoid_dataset.statsCollector.std_velocity}')
        print(f'joint_pos_dataset_mean: {humanoid_dataset.statsCollector.mean_joint_pos}, joint_pos_dataset_std: {humanoid_dataset.statsCollector.std_joint_pos}')
        print(f'ang_vel_dataset_mean: {humanoid_dataset.statsCollector.mean_ang_vel}, ang_vel_dataset_std: {humanoid_dataset.statsCollector.std_ang_vel}')
        print(f'roll_dataset_mean: {humanoid_dataset.statsCollector.mean_roll}, roll_dataset_std: {humanoid_dataset.statsCollector.std_roll}')
        print(f'pitch_dataset_mean: {humanoid_dataset.statsCollector.mean_pitch}, pitch_dataset_std: {humanoid_dataset.statsCollector.std_pitch}')
        print(f'height_dataset_mean: {humanoid_dataset.statsCollector.mean_height}, pitch_dataset_std: {humanoid_dataset.statsCollector.std_height}')
    flow_net = FlowMatchingNet(
        config=config,
        lin_vel_mean=humanoid_dataset.statsCollector.mean_velocity,
        lin_vel_std=humanoid_dataset.statsCollector.std_velocity,
        joint_pos_mean=humanoid_dataset.statsCollector.mean_joint_pos,
        joint_pos_std=humanoid_dataset.statsCollector.std_joint_pos,
        ang_vel_mean=humanoid_dataset.statsCollector.mean_ang_vel,
        ang_vel_std=humanoid_dataset.statsCollector.std_ang_vel,
        roll_mean=humanoid_dataset.statsCollector.mean_roll,
        roll_std=humanoid_dataset.statsCollector.std_roll,
        pitch_mean=humanoid_dataset.statsCollector.mean_pitch,
        pitch_std=humanoid_dataset.statsCollector.std_pitch,
        joint_vel_mean=humanoid_dataset.statsCollector.mean_joint_vel,
        joint_vel_std=humanoid_dataset.statsCollector.std_joint_vel,
        height_mean=humanoid_dataset.statsCollector.mean_height,
        height_std=humanoid_dataset.statsCollector.std_height,
    ).to(dtype=dtype, device=device)
    
    for p in flow_net.text_encoder.parameters():
        p.requires_grad_(False)
        
    flow_net = DDP(
        flow_net,
        device_ids=[local_rank],
        output_device=local_rank,
    )
        
    optimizer = torch.optim.AdamW(flow_net.parameters(), lr=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    optimizer.zero_grad()
    save_folder = 'checkpoints'
    logs_folder = 'logs'
    writer = None
    if is_main_process():
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer = SummaryWriter(f'logs/{cur_time}')
    scaler = torch.amp.GradScaler('cuda', growth_interval=30000)
    # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    epoch_iter = tqdm(range(1000)) if is_main_process() else range(1000)
    for epoch in epoch_iter:
        sampler.set_epoch(epoch)
        # my_schedule = schedule(wait=5, warmup=1, active=12)
        # with profile(activities=activities, schedule=my_schedule) as profilero:
        pbar = (
            tqdm(enumerate(humanoid_dataloader), total=len(humanoid_dataloader))
            if is_main_process()
            else enumerate(humanoid_dataloader)
        )
        for idx, batch in pbar:
            x_1, cond, cu_seqlen_q, cu_seqlen_k, batch_size, max_length_q, max_length_k = batch
            x_1 = x_1.to(device=device, dtype=dtype, non_blocking=True)
            cond = cond.to(device=device, non_blocking=True)
            cu_seqlen_q = cu_seqlen_q.to(device=device, non_blocking=True)
            cu_seqlen_k = cu_seqlen_k.to(device=device, non_blocking=True)
            t = torch.rand(batch_size, 1).to(dtype=dtype, device=device, non_blocking=True)
            t = torch.repeat_interleave(t, cu_seqlen_q[1:] - cu_seqlen_q[:-1], dim=0)
            x_0 = torch.randn_like(x_1)
            x_t = t * x_1 + (1 - t) * x_0
            
            # (output_dim == input_dim)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                u_pred = flow_net(x_t, cond, t, cu_seqlen_q, cu_seqlen_k, max_length_q, max_length_k) # (total_q_len, output_dim)
                loss = torch.mean((u_pred - (x_1 - x_0))**2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # loss_sum += loss.detach()
            if is_main_process() and ((idx + 1) % 100 == 0 or idx == 1):
                writer.add_scalar(
                    "Loss/train", loss.item(),
                    epoch * len(humanoid_dataset) + idx * 32,
                )
                # profilero.step()
        
        scheduler.step()
        # torch.save(flow_net.state_dict(), f'{save_folder}/model_new_weight_{epoch}.pth')
        if is_main_process():
            # Unwrap .module so keys don't carry the "module." prefix.
            os.makedirs(save_folder, exist_ok=True)
            torch.save(
                flow_net.module.state_dict(),
                f"{save_folder}/model_new_weight_{epoch}.pth",
            )
            
        # profilero.export_chrome_trace('trace_gt_0.json')
    
    
if __name__ == '__main__':
    main()
        