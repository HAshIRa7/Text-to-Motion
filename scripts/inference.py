import torch
import torch.nn as nn
import numpy as np
from text_to_motion import FlowMatchingNet, TransformerConfig
from text_to_motion import (
    convert_roll_pitch_ang_vel_to_quat,
    convert_lin_vel_xy_to_root_pos,
)
from datetime import datetime
import os
import math

def edm_schedule(n_points):
    sigma_min = 0.002
    sigma_max = 80.0
    ro = 7.0

    sigma_min_pow = sigma_min**(1/ro)
    sigma_max_pow = sigma_max**(1/ro)

    idx = torch.linspace(0, n_points - 1, n_points - 1) / (n_points - 1)
    sigmas = torch.zeros((n_points))
    sigmas[:-1] = (sigma_max_pow + idx * (sigma_min_pow - sigma_max_pow))**(ro)
    sigmas *= 1 / (sigma_max - sigma_min)
    sigmas -= sigma_min * 1 / (sigma_max - sigma_min)
    sigmas = torch.clamp(sigmas, 0, 1)
    sigmas[-1] = 0
    
    return 1 - sigmas

device = 'cuda:0'
checkpoint_path='checkpoints'
motion_len = 350
n_steps = 200
save_dir = 'generated_motions'
joint_names = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

config = TransformerConfig()
flow_net = FlowMatchingNet(config).to(device)
state_dict = torch.load(checkpoint_path + '/' + 'model_weight_1_19000.pth', weights_only=True)
flow_net.load_state_dict(state_dict)
flow_net.eval()

calculate_parameters = 0
for parameter in flow_net.parameters():
    calculate_parameters += math.prod(parameter.shape)
    
print(f'total net parameters: {calculate_parameters / 10**6}')

print(flow_net)

timesteps = edm_schedule(n_steps + 1).to(device=device)
motion = torch.randn(1, motion_len, config.output_dim).to(device=device)
with torch.no_grad():
    for it in range(n_steps):
        motion = flow_net.midpoint_step(motion, timesteps[it][None], timesteps[it + 1][None])
        
def postprocess_motion(motion: torch.tensor, save_dir: str):
    '''
    return format .npz file with keys
    joint_names - list with str names
    joint_pos - np.ndarray of shape (seq_len, joint_dim)
    body_pos - np.ndarray of shape (seq_len, 1) - root dimension
    body_quat_w - np.ndarray of shape (seq_len, 4) - root quaternion, scalar first
    
    joint_pos - motion[:29]
    roll - motion[29:30]
    pitch - motion[30:31]
    lin_vel - motion[31:33]
    ang_vel - motion[33:]
    '''
    
    height = (motion[0, :, 63:64] * flow_net.height_std[None, :] + flow_net.height_mean[None, :])[:, 0].cpu().numpy()
    ang_vel = (motion[0, :, 33:34] * flow_net.ang_vel_std[None, :] + flow_net.ang_vel_mean[None, :])[:, 0].cpu().numpy()
    roll = (motion[0, :, 29:30] * flow_net.roll_std[None, :] + flow_net.roll_mean[None, :])[:, 0].cpu().numpy()
    pitch = (motion[0, :, 30:31] * flow_net.pitch_std[None, :] + flow_net.pitch_mean[None, :])[:, 0].cpu().numpy()
    quat_w = convert_roll_pitch_ang_vel_to_quat(roll, pitch, ang_vel)[:, None]
    lin_vel = (motion[0, :, 31:33] * flow_net.lin_vel_std[None, :] + flow_net.lin_vel_mean[None, :]).cpu().numpy()
    joint_pos = (motion[0, :, :29] * flow_net.joint_pos_std[None, :] + flow_net.joint_pos_mean[None, :]).cpu().numpy()
    root_pos = convert_lin_vel_xy_to_root_pos(lin_vel, quat_w[:, 0])[:, None]
    root_pos[:, 0, 2] = height
    cur_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        f'{save_dir}/generated_motion_{cur_date}.npz',
        joint_names=joint_names,
        lin_vel=lin_vel,
        joint_pos=joint_pos,
        body_pos_w=root_pos,
        body_quat_w=quat_w,
    )

postprocess_motion(motion, save_dir)
