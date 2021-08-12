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

from save_kitti import SaveKitti

from pathlib import Path
import argparse
import pickle


def save_rgb(obs, filename):
    plt.imshow(obs)
    plt.savefig(filename + ".png")
    plt.clf()


def gen_3d_corners(center, half_extents):
    cx, cy, cz = center
    ex, ey, ez = half_extents
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(np.array([cx + ex * sx, cy + ey * sy, cz + ez * sz, 1]))
    return np.array(corners).T


def corners_to_pixel_pos(points, W, H):
    scale_arr = np.array([[-W/2, 0],[0, H/2]])
    translate_arr = np.array([[W/2, H/2]]*8).T
    return (scale_arr @ points + translate_arr)


def pixel_pos_to_bbox(points):
    max_x, min_x = np.max(points[0]), np.min(points[0])
    max_y, min_y = np.max(points[1]), np.min(points[1])
    # left, top, right, bottom format
    # From https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb
    return np.array([min_x, min_y, max_x, max_y])


def object_to_image_bbox(obj, rgb_obs, sensor_state, sensor_hfov_deg):
    W, H, _ = rgb_obs.shape
    bb = obj.obb
    _, T_camera_world = gen_cam_frame_transform_matrices(sensor_state)
    T_image_camera, _ = gen_cam_projection_matrices(sensor_hfov_deg)

    c = gen_3d_corners(bb.center, bb.half_extents)
    # Convert corners from world frame to camera frame to image space
    c = (T_image_camera @ T_camera_world @ c)
    # Normalize by depth in X, Y, Depth frame to get coordinate on image
    c = c[:2] / c[2]
    c = corners_to_pixel_pos(c, W, H)
    c = pixel_pos_to_bbox(c)
    return c


def gen_second_transformation(num_dim):
    # Coordinate transform puts data in standard right hand rule robot frame. 
    # https://en.wikipedia.org/wiki/Right-hand_rule#Coordinates
    T_second_camera = np.eye(num_dim)
    T_second_camera[[0,1,2]] = T_second_camera[[2,0,1]]
    T_second_camera[0] *= -1
    return T_second_camera
        

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


def transform_second_frame(data):
    return gen_second_transformation(data.shape[0]) @ data


def transform_camera_frame(data, sensor_state):
    x, y, z = data
    _, T_camera_world = gen_cam_frame_transform_matrices(sensor_state)
    arr = T_camera_world @ np.array([[x, y, z, 1]]).T
    return arr.T[0,:3]


def make_bboxes(objs, rgb_obs, sensor_hfov_deg, sensor_state):
    names = []
    bboxes = []
    corners_lst = []
    for obj in objs:
        names.append(obj.category.name())
        bboxes.append(obj.obb)
        corners_lst.append(object_to_image_bbox(obj, rgb_obs, sensor_state, sensor_hfov_deg))
   
    return names, bboxes, corners_lst


def make_pc(depth_obs, T_camera_world, T_second_camera):
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
    # Project to camera frame 3D XYZ from 2.5D pixel X, Pixel Y, Depth, 
    # then project into SECOND frame.
    return T_second_camera @ T_camera_world @ xys

def obj_in_height_range(obj, depth_sensor_state):
    x, y, z = transform_camera_frame(obj.obb.center, depth_sensor_state)
    return (y >= -1 and y <= 1)



def main(dataset_folder, desired_object):
    config=habitat.get_config("task_mp3d.yaml")
    with habitat.Env(
        config=config
    ) as env:
        print("Environment creation successful")
        print("Agent acting inside environment.")
        observations = env.reset()

        save_kitti = SaveKitti(dataset_folder)
        episode_idx = 0
        while episode_idx < 55:
            # observations = env.reset()

            scene = env.sim.semantic_annotations()
            depth_sensor_state = env.sim.get_agent_state().sensor_states["depth"]
            depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

            T_image_camera, T_camera_image = gen_cam_projection_matrices(depth_hfov)
            T_second_camera = gen_second_transformation(4)
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
            num_matching_objects = len(filtered_objects)
            filtered_objects = [obj for obj in filtered_objects if obj_in_height_range(obj, depth_sensor_state)]
            print("Num removed objects:", num_matching_objects - len(filtered_objects))
            if len(filtered_objects) <= 0:
                continue

            print(f"Episode {episode_idx} has {len(filtered_objects)} {desired_object}")
            

            names, bboxes, image_bboxes = make_bboxes(filtered_objects,
                                              observations["rgb"],
                                              depth_hfov,
                                              depth_sensor_state)
            pc = make_pc(observations["depth"], 
                         T_camera_world, 
                         T_second_camera)


            save_kitti.save_instance(episode_idx, 
                                     observations["rgb"], 
                                     names, 
                                     bboxes, 
                                     image_bboxes, 
                                     pc, 
                                     T_second_camera, 
                                     T_camera_world, 
                                     T_camera_image)
            episode_idx += 1
        
        print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SECOND-usable dataset.")
    parser.add_argument('--dataset_folder', default="dataset/training", help="Dataset folder")
    parser.add_argument('--object', default="chair", help="Matterport object name")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder, args.object)
