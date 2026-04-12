import numpy as np

def copysign(mag: float, other: np.ndarray) -> np.ndarray:
    mag_torch = abs(mag) * np.ones_like(other)
    return np.copysign(mag_torch, other)

def convert_quat_to_roll_pitch(quat: np.ndarray):
    '''
    quat of shape: (N, 4)
    '''
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.atan2(sin_roll, cos_roll)
    
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sin_pitch) >= 1, copysign(np.pi / 2.0, sin_pitch), np.asin(sin_pitch))
    
    return roll, pitch

def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    shape = quat.shape
    quat_yaw = quat.reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros_like(quat_yaw)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = quat_yaw / np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat_yaw.reshape(shape)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec + quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qw, qx, qy, qz], axis=-1)