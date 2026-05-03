import os
import numpy as np
from .math import (
    yaw_quat, 
    quat_apply,
    quat_apply_inverse, 
    convert_quat_to_roll_pitch,
    quat_from_euler_xyz,
)
from tqdm.auto import tqdm
import torch

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def convert_lin_vel_to_xy(quat: np.ndarray, lin_vel_w: np.ndarray):
    
    yaw_aligned_quat = yaw_quat(quat)
    lin_vel_yaw_aligned = quat_apply_inverse(yaw_aligned_quat, lin_vel_w)
    
    return lin_vel_yaw_aligned[:, :2]

def convert_roll_pitch_ang_vel_to_quat(roll: np.ndarray, pitch: np.ndarray, ang_vel: np.ndarray, dt: float = 0.02):
    '''
    roll - shape (seq_len,)
    pitch - shape (seq_len,)
    ang_vel - shape (seq_len,)
    '''
    yaw = np.concatenate((np.array([0]), np.cumsum(ang_vel * dt)[:-1]))
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    return quat
    
    
def convert_lin_vel_xy_to_root_pos(lin_vel_yaw_aligned: np.ndarray, quat: np.ndarray, dt: float = 0.02):
    '''
    lin_vel - shape(seq_len, 2)
    '''
    seq_len = lin_vel_yaw_aligned.shape[0]
    # convert lin_vel to world lin_vel, xy
    yaw_aligned_quat = yaw_quat(quat)
    lin_vel_summary = np.zeros(shape=(seq_len, 3))
    lin_vel_summary[:, :2] = lin_vel_yaw_aligned
    world_lin_vel = quat_apply(yaw_aligned_quat, lin_vel_summary)
    root_pos = np.zeros(shape=(seq_len, 3))
    root_pos[:, :2] = np.concatenate(
        (
            np.array([[0.0, 0.0]]),
            np.cumsum(world_lin_vel[:, :2] * dt, axis=0)[:-1]
        )
    )
    root_pos[:, 2] = 0.8
    
    return root_pos

def collect_data(motions_dir: str, motions_len_min: int, motions_len_max: int):
    dct = {}
    for motion_file in tqdm(os.listdir(motions_dir)):
        with np.load(motions_dir +'/' + motion_file, allow_pickle=True) as data:
            motion_len = len(data['joint_pos'])
            num_iterations = motion_len // motions_len_max + ((motion_len % motions_len_max) >= motions_len_min)
            for it in range(num_iterations):
                motion_name = f'{motion_file}_{it}'
                motion_slice = slice(it * motions_len_max, min(it * motions_len_max + motions_len_max, motion_len))
                dct[motion_name] = {}
                dct[motion_name]['text'] = data['text'].item() if 'text' in data else 'A Person ' + ' '.join(motion_file.split('.')[0].split('_'))
                dct[motion_name]['height'] = data['body_pos_w'][motion_slice, 0, 2]
                dct[motion_name]['joint_names'] = list(data['joint_names'])
                dct[motion_name]['joint_pos'] = data['joint_pos'][motion_slice]
                dct[motion_name]['joint_vel'] = data['joint_vel'][motion_slice]
                root_quat_w = data['body_quat_w'][motion_slice, 0]
                roll, pitch = convert_quat_to_roll_pitch(root_quat_w)
                assert roll.shape[0] > 0
                dct[motion_name]['roll'] = roll
                dct[motion_name]['pitch'] = pitch
                dct[motion_name]['velocity'] = convert_lin_vel_to_xy(root_quat_w, data['body_lin_vel_w'][motion_slice, 0])
                dct[motion_name]['ang_vel'] = data['body_ang_vel_w'][motion_slice, 0, 2]
    
    return dct