#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import quaternion

import habitat
import matplotlib.pyplot as plt

# For visualization
import open3d as o3d
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation

from save_data import SaveData

from pathlib import Path
import argparse


def gen_second_transformation(num_dim):
    # Coordinate transform puts data in standard right hand rule robot frame. 
    # https://en.wikipedia.org/wiki/Right-hand_rule#Coordinates
    T_second_camera = np.eye(num_dim)
    T_second_camera[[0,1,2]] = T_second_camera[[2,0,1]]
    T_second_camera[0] *= -1
    T_camera_second = np.linalg.inv(T_second_camera)
    return T_second_camera, T_camera_second
        

def gen_cam_projection_matrices(sensor_hfov_deg):
    hfov = float(sensor_hfov_deg) * np.pi / 180.0

    # Camera matrix constructed using pinhole camera model and assumption that W==H.
    T_image_camera = np.array([
        [1 / np.tan(hfov / 2.), 0.0,                   0.0, 0.0],
        [0.0,                   1 / np.tan(hfov / 2.), 0.0, 0.0],
        [0.0,                   0.0,                   1.0, 0.0],
        [0.0,                   0.0,                   0.0, 1.0]
    ])
    T_camera_image = np.linalg.inv(T_image_camera) # Converts image space coords to 3D camera frame coords.
    return T_image_camera, T_camera_image


def gen_cam_frame_transform_matrices(sensor_state):
    # Construct transform from camera frame to world frame using camera position info.
    T_world_camera = np.eye(4)
    T_world_camera[0:3,0:3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    T_world_camera[0:3,3] = sensor_state.position
    # Invert to get transform from world frame to camera frame.
    T_camera_world = np.linalg.inv(T_world_camera)
    return T_world_camera, T_camera_world


def make_pc(depth_obs):
    # From https://aihabitat.org/docs/habitat-api/view-transform-warp.html
    # Assumes width and height are same for simplicity.
    W, H, d = depth_obs.shape
    assert(W == H)
    assert(d == 1)

    depth_obs = depth_obs.reshape(W, H)

    # Sample uniformly points in grid.
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up.
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    depth = depth_obs.reshape(1, W, H)
    xs = xs.reshape(1, W, H)
    ys = ys.reshape(1, W, H)
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)  # Flatten to 4 by (W*H) matrix.
    # Pixel X, Pixel Y, Depth
    return xys


def main(dataset_folder, desired_object):
    config=habitat.get_config("task_mp3d.yaml")
    with habitat.Env(
        config=config
    ) as env:
        print("Environment creation successful")
        print("Agent acting inside environment.")
        observations = env.reset()

        save_kitti = SaveData(dataset_folder)
        episode_idx = 0
        while episode_idx < 55:
            # observations = env.reset()

            scene = env.sim.semantic_annotations()
            depth_sensor_state = env.sim.get_agent_state().sensor_states["depth"]
            depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

            T_image_camera, T_camera_image = gen_cam_projection_matrices(depth_hfov)
            T_second_camera, T_camera_second = gen_second_transformation(4)
            T_world_camera, T_camera_world = gen_cam_frame_transform_matrices(depth_sensor_state)

            # Each id in the semantic image is an instance ID, which means we can lookup 
            # the actual *object* from the scene.
            # https://aihabitat.org/docs/habitat-sim/habitat_sim.scene.SemanticObject.html
            # Idea from https://github.com/facebookresearch/habitat-sim/issues/263#issuecomment-537295069            
            instance_id_to_obj = {int(obj.id.split("_")[-1]): obj for obj in scene.objects}
            # Extract the unique objects, get their Object Bounding Boxes.
            # https://aihabitat.org/docs/habitat-sim/habitat_sim.geo.OBB.html
            distinct_objects = [instance_id_to_obj[instance_id] for instance_id in set(observations["semantic"].flatten())]

            filtered_objects = [obj for obj in distinct_objects if obj.category.name() == desired_object]            
            if len(filtered_objects) <= 0:
                print("No such objects found in scene")
                continue
            
            names = [obj.category.name() for obj in filtered_objects]
            bboxes = [obj.obb for obj in filtered_objects] 
            pc = make_pc(observations["depth"])
            num_saved_boxes = save_kitti.save_instance(episode_idx, 
                                                       observations["rgb"], 
                                                       names, 
                                                       bboxes, 
                                                       pc, 
                                                       T_second_camera, 
                                                       T_camera_second, 
                                                       T_camera_world, 
                                                       T_image_camera)

            if num_saved_boxes <= 0:
                print("No boxes met filter criteria")
                continue

            print(f"Episode {episode_idx} has {len(filtered_objects)} of {desired_object}")
            episode_idx += 1
        
        print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SECOND-usable dataset.")
    parser.add_argument('--dataset_folder', default="dataset/training", help="Dataset folder")
    parser.add_argument('--object', default="chair", help="Matterport object name")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder, args.object)
