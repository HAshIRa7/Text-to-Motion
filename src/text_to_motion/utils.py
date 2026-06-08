import os
from pathlib import Path
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
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel
import traceback
import pickle

NEXT_WORDS = ["After", "Next", "Then", "Consequently"]
FPS = 50

g1_data_names2size = {
    'velocity': 2,
    'joint_pos': 29,
    'joint_vel': 29,
    'ang_vel': 1,
    'roll': 1,
    'pitch': 1,
    'height': 1,
}

class StatisticCollector:
    def __init__(self,):
        pass

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

def load_file(motions_dir: str, motions_new_dir, motion_file: str, motions_len_min: int, motions_len_max: int) -> StatisticCollector:
    try:
        localstatsCollector = StatisticCollector()
        for k, siz in g1_data_names2size.items():
            setattr(localstatsCollector, f'mean_{k}', torch.zeros(siz))
            setattr(localstatsCollector, f'mean_{k}_squared', torch.zeros(siz))
            setattr(localstatsCollector, f'std_{k}', torch.ones(siz))
            setattr(localstatsCollector, f'local_len', 0)
            
        tmp_dict = {}
        with np.load(os.path.join(motions_dir, motion_file), allow_pickle=True) as data:
            metadata = data['metadata']
            new_metadata = [item for item in metadata]
            for offset in range(1, len(metadata)):
                for i in range(len(metadata) - offset):
                    prompt = metadata[i]['description']
                    for j in range(i + 1, i + offset + 1):
                        prompt += NEXT_WORDS[random.randint(0, len(NEXT_WORDS) - 1)] + ' ' + metadata[j]['description']
                    
                    new_metadata.append({
                        'start_time': metadata[i]['start_time'],
                        'end_time': metadata[i + offset]['end_time'],
                        'description': prompt
                    })
                    
            tmp_dict['height'] = data['body_pos_w'][:, 0, 2]
            tmp_dict['joint_pos'] = data['joint_pos']
            tmp_dict['joint_vel'] = data['joint_vel']
            root_quat_w = data['body_quat_w'][:, 0]
            roll, pitch = convert_quat_to_roll_pitch(root_quat_w)
            assert roll.shape[0] > 0
            tmp_dict['roll'] = roll
            tmp_dict['pitch'] = pitch
            tmp_dict['velocity'] = convert_lin_vel_to_xy(root_quat_w, data['body_lin_vel_w'][:, 0])
            tmp_dict['ang_vel'] = data['body_ang_vel_w'][:, 0, 2]
            
            localstatsCollector.local_len += len(roll)
                    
            for k in g1_data_names2size:
                aggregated_first_momentum = np.sum(tmp_dict[k], axis=0)
                aggregated_second_momentum = np.sum(tmp_dict[k]**2, axis=0) 
                if not isinstance(aggregated_first_momentum, np.ndarray):
                    aggregated_first_momentum = aggregated_first_momentum[None]
                    aggregated_second_momentum = aggregated_second_momentum[None]
                setattr(localstatsCollector, f'mean_{k}', getattr(localstatsCollector, f'mean_{k}') + torch.from_numpy(aggregated_first_momentum))
                setattr(localstatsCollector, f'mean_{k}_squared', getattr(localstatsCollector, f'mean_{k}_squared') + torch.from_numpy(aggregated_second_momentum))
                        
            motion_len_total = len(data['joint_pos'])
            for it, one_metadata in enumerate(new_metadata):
                motion_start_fps = min(int(one_metadata['start_time'] * FPS), motion_len_total - 1)
                motion_end_fps = min(int(one_metadata['end_time'] * FPS), motion_len_total - 1)
                assert motion_end_fps - motion_start_fps > 0
                motion_len = motion_end_fps - motion_start_fps
                num_iterations = motion_len // motions_len_max + ((motion_len % motions_len_max) >= motions_len_min)
                for new_it in range(num_iterations):
                    dct = {}
                    motion_name = f'{motion_file.split(".")[0]}_{it}_{new_it}'
                    motion_slice = slice(motion_start_fps + new_it * motions_len_max, min(motion_start_fps + (new_it + 1) * motions_len_max, motion_end_fps))
                    dct[motion_name] = {}
                    dct[motion_name]['text'] = one_metadata['description']
                    dct[motion_name]['height'] = data['body_pos_w'][motion_slice, 0, 2]
                    dct[motion_name]['joint_names'] = list(data['joint_names'])
                    dct[motion_name]['joint_pos'] = data['joint_pos'][motion_slice]
                    dct[motion_name]['joint_vel'] = data['joint_vel'][motion_slice]
                    root_quat_w = data['body_quat_w'][motion_slice, 0]
                    roll, pitch = convert_quat_to_roll_pitch(root_quat_w)
                    assert roll.shape[0] > 0
                    assert roll.shape[0] <= motions_len_max
                    dct[motion_name]['roll'] = roll
                    dct[motion_name]['pitch'] = pitch
                    dct[motion_name]['velocity'] = convert_lin_vel_to_xy(root_quat_w, data['body_lin_vel_w'][motion_slice, 0])
                    dct[motion_name]['ang_vel'] = data['body_ang_vel_w'][motion_slice, 0, 2]
                    
                    np.savez(
                        os.path.join(motions_new_dir, motion_name),
                        velocity=dct[motion_name]['velocity'],
                        joint_vel=dct[motion_name]['joint_vel'],
                        joint_pos=dct[motion_name]['joint_pos'],
                        roll=dct[motion_name]['roll'],
                        pitch=dct[motion_name]['pitch'],
                        ang_vel=dct[motion_name]['ang_vel'],
                        height=dct[motion_name]['height'],
                        text=np.array(dct[motion_name]['text'], dtype=object),
                    )
    except Exception as e:
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Repr: {repr(e)}")
        print("Full Traceback:")
        traceback.print_exc()
        print(f'Exception: {e} on file {motion_file}')
                
    return localstatsCollector

def collect_data(motions_dir: str, motions_new_dir: str, motions_len_min: int, motions_len_max: int) -> StatisticCollector:
    statsCollector = StatisticCollector()
    for k, siz in g1_data_names2size.items():
        setattr(statsCollector, f'mean_{k}', torch.zeros(siz))
        setattr(statsCollector, f'mean_{k}_squared', torch.zeros(siz))
        setattr(statsCollector, f'std_{k}', torch.ones(siz))
        
    pure_motions_len = 0
    Path(motions_new_dir).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(load_file, motions_dir, motions_new_dir, motion_file, motions_len_min, motions_len_max): motion_file for motion_file in os.listdir(motions_dir)}
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                localstatsCollector = future.result()
                pure_motions_len += localstatsCollector.local_len
                for k in g1_data_names2size:
                    setattr(statsCollector, f'mean_{k}', getattr(statsCollector, f'mean_{k}') + getattr(localstatsCollector, f'mean_{k}'))
                    setattr(statsCollector, f'mean_{k}_squared', getattr(statsCollector, f'mean_{k}_squared') + getattr(localstatsCollector, f'mean_{k}_squared'))
                pbar.update(1)
    
    for k in g1_data_names2size:
        setattr(statsCollector, f'mean_{k}', getattr(statsCollector, f'mean_{k}') / pure_motions_len)
        setattr(statsCollector, f'mean_{k}_squared', getattr(statsCollector, f'mean_{k}_squared') / pure_motions_len)
        setattr(statsCollector, f'std_{k}', torch.sqrt(getattr(statsCollector, f'mean_{k}_squared') - getattr(statsCollector, f'mean_{k}')**2))
    
    with open('statistic_collector.pkl', 'wb') as out_file:
        pickle.dump(statsCollector, out_file, pickle.HIGHEST_PROTOCOL)
    return statsCollector