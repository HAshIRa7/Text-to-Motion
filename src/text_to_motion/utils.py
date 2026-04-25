import os
import numpy as np
from .math import (
    yaw_quat, 
    quat_apply,
    quat_apply_inverse, 
    convert_quat_to_roll_pitch,
    quat_from_euler_xyz,
)

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

def collect_data(motions_dir: str, motions_len: int):
    dct = {}
    for motion_file in os.listdir(motions_dir):
        with np.load(motions_dir +'/' + motion_file, allow_pickle=True) as data:
            if len(data['joint_pos']) < motions_len:
                continue
            dct[motion_file] = {}
            dct[motion_file]['height'] = data['body_pos_w'][:, 0, 2]
            dct[motion_file]['joint_names'] = list(data['joint_names'])
            dct[motion_file]['joint_pos'] = data['joint_pos']
            dct[motion_file]['joint_vel'] = data['joint_vel']
            root_quat_w = data['body_quat_w'][:, 0]
            roll, pitch = convert_quat_to_roll_pitch(root_quat_w)
            dct[motion_file]['roll'] = roll
            dct[motion_file]['pitch'] = pitch
            dct[motion_file]['lin_vel'] = convert_lin_vel_to_xy(root_quat_w, data['body_lin_vel_w'][:, 0])
            dct[motion_file]['ang_vel'] = data['body_ang_vel_w'][:, 0, 2]
    
    return dct