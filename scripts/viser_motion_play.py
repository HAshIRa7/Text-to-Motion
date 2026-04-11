from __future__ import annotations

import time
from typing import Literal, List
import os

import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf

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

def main(
    motion_folder: str = 'generated_motions',
    robot_type = "g1",
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    # Start viser server.
    server = viser.ViserServer()
    server.initial_camera.position = (1.2, 1.2, 1.2)
    
    reset = False 

    dct = {}
    motions_files = os.listdir(motion_folder)
    for motion_file in motions_files:
        dct[motion_file] = {}
        with np.load(motion_folder + '/' + motion_file, allow_pickle=True) as data:
            dct[motion_file]['joint_names'] = list(data['joint_names'])
            dct[motion_file]['joint_pos'] = data['joint_pos']
            dct[motion_file]['body_pos_w'] = data['body_pos_w']
            dct[motion_file]['body_quat_w'] = data['body_quat_w']
    
        dct[motion_file]['motion_len'] = len(dct[motion_file]['joint_pos'])
    
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
    
    for motion_file in motions_files:
        dct[motion_file]['joint_pos'] = match_joint_names(
            joint_names_data=dct[motion_file]['joint_names'], 
            urdf_joint_names=list(viser_urdf.get_actuated_joint_limits()), 
            joint_data=dct[motion_file]['joint_pos'],
        )
    
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

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Create grid.
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
    
    start_motion = server.gui.add_button("Start Motion")
    
    @start_motion.on_click
    def _(_):
        nonlocal reset
        reset = False
        cur_motion = dropdown.value
        it = 0
        while it < dct[cur_motion]['motion_len']:
            if dropdown.value != cur_motion:
                it = 0
                cur_motion = dropdown.value
            if reset:
                robot_base.position = (0.0, 0.0, 0.0)
                robot_base.wxyz = (1.0, 0.0, 0.0, 0.0)
                viser_urdf.update_cfg(np.zeros(len(viser_urdf.get_actuated_joint_limits())))
                break
            
            cur_joint_pos = dct[cur_motion]['joint_pos'][it]
            cur_root_pos_w = dct[cur_motion]['body_pos_w'][it][0]
            cur_orientation_w = dct[cur_motion]['body_quat_w'][it][0]
            
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
        
    dropdown = server.gui.add_dropdown(label="Motions", options=motions_files)
    
    gui_text = server.gui.add_text(
                "Text",
                initial_value="Hello world",
            )

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)