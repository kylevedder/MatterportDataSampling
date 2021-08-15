#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.type_check import imag
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


def make_pc(depth_obs,
            T_robot_camera,
            height_cutoff=2,
            subsample_rows=10,
            subsample_rate=5):
    # From https://aihabitat.org/docs/habitat-api/view-transform-warp.html
    # Assumes width and height are same for simplicity.
    W, H, d = depth_obs.shape
    assert (W == H)
    assert (d == 1)

    depth_obs = depth_obs.reshape(W, H)

    # Sample uniformly points in grid.
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up.
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    depth = depth_obs.reshape(1, H, W)
    xs = xs.reshape(1, H, W)
    ys = ys.reshape(1, H, W)
    xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)  # Flatten to 4 by (W*H) matrix.

    # 4 by N points
    pc = T_robot_camera @ xys
    # N by 4 points
    pc = pc.T.astype(np.float32)
    # N by 3 points: X, Y, Z
    pc = pc[:, :3]
    # Convert back to image shape for subsampling of rows
    pc = pc.T.reshape(3, H, W)
    image_pc = np.transpose(pc, (1, 2, 0))
    pc = pc[:, 0::subsample_rows]
    # N by 3 points
    pc = pc.reshape(3, -1).T

    # Subsample pc.
    pc = pc[0::subsample_rate]
    # Chop below height cutoff.
    pc = pc[pc[:, 2] < height_cutoff]

    return pc, image_pc


def is_in_range(obb, T_robot_world, max_distance=7):
    center = (T_robot_world @ np.array([*obb.center, 1]))[:3]
    x, y, z = center
    if z < -1 or z > 1:
        print("outside Z", z)
        return False
    xynorm = np.linalg.norm(center[:2], 2)
    if xynorm > max_distance:
        print("object center dist", xynorm, "outside max dist", max_distance)
        return False
    return True


def make_bboxes(semantic_obs,
                image_pc,
                scene,
                desired_objects,
                T_robot_world,
                min_points_per_obj=50):
    assert semantic_obs.shape == image_pc.shape[:2]
    if type(desired_objects) is not list:
        desired_objects = [desired_objects]
    # Each id in the semantic image is an instance ID, which means we can lookup
    # the actual *object* from the scene.
    # https://aihabitat.org/docs/habitat-sim/habitat_sim.scene.SemanticObject.html
    # Idea from https://github.com/facebookresearch/habitat-sim/issues/263#issuecomment-537295069
    instance_id_to_obj = {
        int(obj.id.split("_")[-1]): obj
        for obj in scene.objects
    }
    # Extract the unique objects, get their Object Bounding Boxes.
    # https://aihabitat.org/docs/habitat-sim/habitat_sim.geo.OBB.html
    uniques, counts = np.unique(semantic_obs.flatten(), return_counts=True)
    # Remove objects with fewer than `min_points_per_obj` observable points.
    uniques = uniques[counts > min_points_per_obj]
    distinct_objects = [
        instance_id_to_obj[instance_id] for instance_id in uniques
    ]

    filtered_objects = [
        obj for obj in distinct_objects
        if obj.category.name() in desired_objects
    ]
    filtered_objects = [
        obj for obj in filtered_objects if is_in_range(obj.obb, T_robot_world)
    ]
    if len(filtered_objects) <= 0:
        print("No such objects found in scene")
        return [], None

    names = [obj.category.name() for obj in filtered_objects]
    bboxes = [obj.obb for obj in filtered_objects]
    return names, bboxes


def main(dataset_folder, desired_object, num_scenes):
    config = habitat.get_config("task_mp3d.yaml")
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        print("Agent acting inside environment.")
        observations = env.reset()

        save_kitti = SaveData(dataset_folder)
        episode_idx = 0
        while episode_idx < num_scenes:
            observations = env.reset()

            scene = env.sim.semantic_annotations()
            depth_sensor_state = env.sim.get_agent_state(
            ).sensor_states["depth"]
            depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

            T_image_camera, T_camera_image = gen_cam_projection_matrices(
                depth_hfov)
            T_robot_camera, T_camera_robot = gen_robot_transformation(4)
            T_world_camera, T_camera_world = gen_cam_frame_transform_matrices(
                depth_sensor_state)

            pc, image_pc = make_pc(observations["depth"], T_robot_camera)
            names, bboxes = make_bboxes(observations["semantic"], image_pc,
                                        scene, desired_object,
                                        T_robot_camera @ T_camera_world)
            if bboxes is None:
                continue

            num_saved_boxes = save_kitti.save_instance(
                episode_idx, observations["rgb"], names, bboxes, pc,
                T_robot_camera, T_camera_robot, T_camera_world, T_image_camera)

            if num_saved_boxes <= 0:
                print("No boxes met filter criteria")
                continue

            print(
                f"Episode {episode_idx} has {num_saved_boxes} of {desired_object}"
            )
            episode_idx += 1

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Open3D's PointPillars-usable dataset.")
    parser.add_argument('--dataset_folder',
                        default="dataset/training",
                        help="Dataset folder")
    parser.add_argument('--object',
                        default="chair",
                        help="Matterport object name")
    parser.add_argument('--num_scenes',
                        type=int,
                        default=7500,
                        help="Number of scenes")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder, args.object, args.num_scenes)
