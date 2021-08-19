#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.type_check import imag
import pandas as pd
import quaternion
import glob

import habitat
import matplotlib.pyplot as plt

# For visualization
import open3d as o3d
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation

from save_data import SaveData

from pathlib import Path
import argparse

from make_pc import make_pc
from make_bboxes import make_bboxes


def gen_robot_transformation(num_dim):
    # Coordinate transform puts data in standard right hand rule robot frame.
    # https://en.wikipedia.org/wiki/Right-hand_rule#Coordinates
    T_robot_camera = np.eye(num_dim)
    T_robot_camera[[0, 1, 2]] = T_robot_camera[[2, 0, 1]]
    T_robot_camera[0] *= -1
    T_camera_robot = np.linalg.inv(T_robot_camera)
    return T_robot_camera, T_camera_robot


def gen_cam_projection_matrices(sensor_hfov_deg):
    hfov = float(sensor_hfov_deg) * np.pi / 180.0

    # Camera matrix constructed using pinhole camera model and assumption that W==H.
    T_image_camera = np.array([[1 / np.tan(hfov / 2.), 0.0, 0.0, 0.0],
                               [0.0, 1 / np.tan(hfov / 2.), 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    T_camera_image = np.linalg.inv(
        T_image_camera
    )  # Converts image space coords to 3D camera frame coords.
    return T_image_camera, T_camera_image


def gen_cam_frame_transform_matrices(sensor_state):
    # Construct transform from camera frame to world frame using camera position info.
    T_world_camera = np.eye(4)
    T_world_camera[0:3,
                   0:3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    T_world_camera[0:3, 3] = sensor_state.position
    # Invert to get transform from world frame to camera frame.
    T_camera_world = np.linalg.inv(T_world_camera)
    return T_world_camera, T_camera_world


def get_split_configs(split):
    if split == "training":
        return sorted(list(glob.glob("task_mp3d_train*")))
    elif split == "test":
        return sorted(list(glob.glob("task_mp3d_test*")))
    raise FileNotFoundError(f"Unknown split: {split}")


def main(dataset_folder, desired_object, num_scenes, save_vis, split):
    split_configs = get_split_configs(split)
    num_episodes_per_split = num_scenes // len(split_configs)

    save_data = SaveData(dataset_folder)
    total_episode_idx = 0

    for config_file in split_configs:
        config = habitat.get_config(config_file)
        with habitat.Env(config=config) as env:
            print("Environment creation successful")
            print("Agent acting inside environment.")
            episode_idx = 0
            while episode_idx < num_episodes_per_split:
                observations = env.reset()

                scene = env.sim.semantic_annotations()
                depth_sensor_state = env.sim.get_agent_state().sensor_states["depth"]
                depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

                T_image_camera, T_camera_image = gen_cam_projection_matrices(
                    depth_hfov)
                T_robot_camera, T_camera_robot = gen_robot_transformation(4)
                T_world_camera, T_camera_world = gen_cam_frame_transform_matrices(
                    depth_sensor_state)

                pc, image_pc = make_pc(observations["depth"], T_robot_camera)
                names, bboxes = make_bboxes(observations["semantic"], image_pc, scene,
                                            desired_object,
                                            T_robot_camera @ T_camera_world,
                                            T_image_camera @ T_camera_robot)
                # if len(bboxes) <= 0:
                #     continue

                # plt.imshow(observations["rgb"])
                # print(f"Image name: img{total_episode_idx:06d}.png", config_file)
                # plt.savefig(f"img{total_episode_idx:06d}.png")
                # plt.clf()

                save_data.save_instance(total_episode_idx, names, bboxes, pc, T_camera_world,
                                        T_image_camera, save_vis)

                print(f"Episode {total_episode_idx} has {len(bboxes)}")
                episode_idx += 1
                total_episode_idx += 1

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Open3D's PointPillars-usable dataset.")
    parser.add_argument('--dataset_folder',
                        default="dataset",
                        help="Dataset folder")
    parser.add_argument('--split', default="training", help="Number of scenes")
    parser.add_argument('--object',
                        default="chair",
                        help="Matterport object name")
    parser.add_argument('--num_scenes',
                        type=int,
                        default=7500,
                        help="Number of scenes")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--savevis",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="Save visualization.")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder + "/" + args.split, args.object, args.num_scenes,
         args.savevis, args.split)
