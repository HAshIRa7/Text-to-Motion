from __future__ import annotations

import time
from typing import Literal, List
import os
import torch

import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description
from text_to_motion import (
    FlowMatchingNet, 
    TransformerConfig, 
    last_token_pool,
    convert_roll_pitch_ang_vel_to_quat,
    convert_lin_vel_xy_to_root_pos,
)
from transformers import AutoTokenizer, AutoModel

import viser
from viser.extras import ViserUrdf

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

def postprocess_motion(flow_net: torch.nn.Module, motion: torch.tensor):
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
    motion = motion.float()
    height = (motion[0, :, 63:64] * flow_net.height_std[None, :] + flow_net.height_mean[None, :])[:, 0].cpu().numpy()
    ang_vel = (motion[0, :, 33:34] * flow_net.ang_vel_std[None, :] + flow_net.ang_vel_mean[None, :])[:, 0].cpu().numpy()
    roll = (motion[0, :, 29:30] * flow_net.roll_std[None, :] + flow_net.roll_mean[None, :])[:, 0].cpu().numpy()
    pitch = (motion[0, :, 30:31] * flow_net.pitch_std[None, :] + flow_net.pitch_mean[None, :])[:, 0].cpu().numpy()
    quat_w = convert_roll_pitch_ang_vel_to_quat(roll, pitch, ang_vel)[:, None]
    lin_vel = (motion[0, :, 31:33] * flow_net.lin_vel_std[None, :] + flow_net.lin_vel_mean[None, :]).cpu().numpy()
    joint_pos = (motion[0, :, :29] * flow_net.joint_pos_std[None, :] + flow_net.joint_pos_mean[None, :]).cpu().numpy()
    root_pos = convert_lin_vel_xy_to_root_pos(lin_vel, quat_w[:, 0])[:, None]
    root_pos[:, 0, 2] = height
    return {
        'joint_names': joint_names,
        'lin_vel': lin_vel,
        'joint_pos': joint_pos,
        'body_pos_w': root_pos,
        'body_quat_w': quat_w,
    }

class InferenceModel:
    
    def __init__(self, checpoint_path: str, diffusion_steps: int = 100, device: str = 'cuda', dtype: torch.dtype = torch.float32): 
        self.config = TransformerConfig()
        flow_net = FlowMatchingNet(self.config)
        state_dict = torch.load(checpoint_path, weights_only=True)
        flow_net.load_state_dict(state_dict)
        self.flow_net = flow_net.to(device=device, dtype=dtype)
        self.flow_net.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B').to(device)
        
        self.schedule = edm_schedule(diffusion_steps + 1).to(device=device, dtype=dtype)
        self.motion_len = int(7.0 * 50)
        
        self.device=device
        self.dtype=dtype
        self.diffusion_steps = diffusion_steps
        
    def generate(self, text: str):
        
        batch = [text, '']
        max_length = 8192
        batch_dict = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():   
            outputs = self.model(**batch_dict)
            embed = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to(dtype=self.dtype)
        cond_embed = embed[0:1]
        uncond_embed = embed[1:2]
        motion = torch.randn(1, self.motion_len, self.config.output_dim).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for it in range(self.diffusion_steps):
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    motion = self.flow_net.midpoint_step(motion, cond_embed, self.schedule[it][None], self.schedule[it + 1][None])
                    # motion = self.flow_net.guidance_step(motion, cond_embed, uncond_embed, self.schedule[it][None], self.schedule[it + 1][None])
        
        return postprocess_motion(self.flow_net, motion)

def match_joint_names(joint_names_data: List[str], urdf_joint_names: List[str], joint_data: np.ndarray):
    '''
    joint_names_data: order of joints in data
    urdf_joint_names: order of joints in urdf
    joints_data: np.array of shape (motion_len, num_joints)
    
    return: joints data in urdf order
    '''
    permutation = []
    for urdf_joint_name in urdf_joint_names:
        permutation.append(joint_names_data.index(urdf_joint_name))
    
    return joint_data[:, permutation]

# prev 1_11000
def main(
    robot_type = "g1",
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    checkpoint_path: str = 'checkpoints/model_weight_2_8000.pth',
) -> None:
    # Start viser server.
    server = viser.ViserServer()
    server.initial_camera.position = (1.2, 1.2, 1.2)
    
    reset = False
    
    inference = InferenceModel(
        checpoint_path=checkpoint_path,
        diffusion_steps=200,
        device='cuda:0',
        dtype=torch.float32,
    )
    
    robot_base = server.scene.add_frame("/robot", show_axes=False)
    
    urdf = load_robot_description(
        robot_type + "_description",
        load_meshes=load_meshes,
        build_scene_graph=load_meshes,
        load_collision_meshes=load_collision_meshes,
        build_collision_scene_graph=load_collision_meshes,
    )
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        root_node_name="/robot",
    )
    
    viser_urdf.update_cfg(np.zeros(len(viser_urdf.get_actuated_joint_limits())))
    
    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            0.0,
        ),
    )
    
    generate_motion = server.gui.add_button("Generate Motion")
    
    @generate_motion.on_click
    def _(_):
        nonlocal reset
        reset = False
        cur_prompt = gui_text.value 
        motion = inference.generate(cur_prompt)
        it = 0
        while it < inference.motion_len:
            if reset:
                robot_base.position = (0.0, 0.0, 0.0)
                robot_base.wxyz = (1.0, 0.0, 0.0, 0.0)
                viser_urdf.update_cfg(np.zeros(len(viser_urdf.get_actuated_joint_limits())))
                break
            
            cur_joint_pos = motion['joint_pos'][it]
            cur_root_pos_w = motion['body_pos_w'][it][0]
            cur_orientation_w = motion['body_quat_w'][it][0]
            
            time.sleep(0.02)
            
            viser_urdf.update_cfg(cur_joint_pos)
            robot_base.position = cur_root_pos_w
            robot_base.wxyz = cur_orientation_w
            
            it += 1
            

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        nonlocal reset
        reset = True
    
    gui_text = server.gui.add_text(
                "Text",
                initial_value="A person walks forward",
            )

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)