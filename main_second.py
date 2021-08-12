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


def save_bboxes(objs, rgb_obs, sensor_hfov_deg, sensor_state, filename, save_bbox_img=False):
    mesh = o3d.geometry.TriangleMesh()

    W, H, d = rgb_obs.shape
    assert(d == 3)
        
    corners_lst = []
    # Insert objects in world frame
    for obj in objs:
        bb = obj.obb
        box = o3d.geometry.TriangleMesh.create_box(*(bb.half_extents * 2))
        r1, r2, r3, r4 = bb.rotation
        box.rotate(Rotation.from_quat([r1, r2, r3, r4]).as_matrix(), 
                box.get_center())
        xt, yt, zt = bb.center
        box.translate((xt, yt, zt), relative=False)
        # cyl.translate(-sensor_state.position, relative=True)
        mesh += box
        corners_lst.append(object_to_image_bbox(obj, rgb_obs, sensor_state, sensor_hfov_deg))
    
    # Convert mesh from world frame to camera frame
    _, T_camera_world = gen_cam_frame_transform_matrices(sensor_state)
    # Convert mesh from camera frame to SECOND frame
    T_second_camera = gen_second_transformation(4)

    # Put mesh in camera frame
    mesh.transform(T_camera_world)
    mesh.transform(T_second_camera)
    o3d.io.write_triangle_mesh(filename+".ply", mesh)

    if save_bbox_img:
        plt.imshow(rgb_obs)
        for cs in corners_lst:
            left, top, right, bottom = cs
            width = right - left
            height = bottom - top
            rect = plt.Rectangle((left, top), width, height,
                        facecolor="green", alpha=0.5)
            plt.gca().add_patch(rect)
        plt.savefig(filename + ".jpg")


def save_pc(sensor_hfov_deg, depth_obs, filename):
    # From https://aihabitat.org/docs/habitat-api/view-transform-warp.html
    # Assumes width and height are same for simplicity.
    W, H, d = depth_obs.shape
    assert(W == H)
    assert(d == 1)

    depth_obs = depth_obs.reshape(W, H)

    # Converts image space coords to 3D camera frame coords.
    _, T_camera_world = gen_cam_projection_matrices(sensor_hfov_deg)
    # Converts 3D camera frame coords into SECOND's frame (standard robot frame).
    T_second_camera = gen_second_transformation(4)
    

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
    xy_c0 = T_second_camera @ T_camera_world @ xys

    PyntCloud(pd.DataFrame(data=xy_c0[:3].T,
        columns=["x", "y", "z"])).to_file(filename + ".ply")

    # Save in KITTI PC binary format of X, Y, Z, Intensity, with Intensity always 1.
    xy_c0[3] = 1
    with open(filename + ".bin", "wb") as bin_f:
        bin_f.write(xy_c0.T.ravel().astype(np.float32))

def obj_in_height_range(obj, depth_sensor_state):
    x, y, z = transform_camera_frame(obj.obb.center, depth_sensor_state)
    return (y >= -1 and y <= 1)

    
def gen_entries(sensor_state, idx, db_entries_so_far, rgb_file, pc_file, rgb_obs, objects, desired_object, depth_sensor_state, sensor_hfov_deg):
    info_entry = {
    "image" : {
        "image_idx" : idx,
        "image_path" : rgb_file + ".png",
        "image_shape" : np.array(list(rgb_obs.shape)[:2])
    },
    "calib" : {
        "P0" : np.eye(4),
        "P1" : np.eye(4),
        "P2" : np.eye(4),
        "P3" : np.eye(4),
        "R0_rect" : np.eye(4),
        "Tr_velo_to_cam" : np.eye(4),
        "Tr_imu_to_velo" : np.eye(4)
    },
    "point_cloud" : {
        "num_features" : 4,
        "velodyne_path" : pc_file + ".bin"
    },
    "annos" : {
        "name" : np.array([o.category.name() for o in objects]),
        "truncated" : np.zeros(len(objects)),
        "occluded" : np.zeros(len(objects)),
        "alpha" : np.ones(len(objects)) * -10,
        "bbox" : np.array([object_to_image_bbox(o, rgb_obs, depth_sensor_state, sensor_hfov_deg) for o in objects]),
        "dimensions" : np.array([o.obb.half_extents * 2 for o in objects]),
        "location" : np.array([transform_second_frame(transform_camera_frame(o.obb.center, sensor_state)) for o in objects]),
        "rotation_y" : np.array([Rotation.from_quat(o.obb.rotation).as_euler('xyz',degrees=True)[0] for o in objects]), # Use the 0th element because X is Y in SECOND coords
        "score" : np.ones(len(objects)),
        "index" : np.array(range(len(objects))),
        "group_ids" : np.array(range(len(objects))),
        "difficulty" : np.zeros(len(objects)),
        "num_points_in_gt" : np.ones(len(objects)) * 999
    }
    }

    extract_y = lambda rot: Rotation.from_quat(rot).as_euler('yxz', degrees=True)[:1]
    dbinfo_obj_entries = [{
            "name" : desired_object,
            "path" : pc_file + ".bin",
            "image_idx" : idx,
            "gt_idx" : i + db_entries_so_far,
            "box3d_lidar" : np.concatenate([o.obb.center, o.obb.half_extents * 2, extract_y(o.obb.rotation)]),
            "num_points_in_gt": 999, 
            "difficulty": 0, 
            "group_id": 0
    } for i, o in enumerate(objects)]
    return info_entry, dbinfo_obj_entries


def main(dataset_folder, desired_object):
    config=habitat.get_config("task_mp3d.yaml")
    with habitat.Env(
        config=config
    ) as env:
        print("Environment creation successful")
        print("Agent acting inside environment.")
        episode_idx = 0
        info_entries = []
        dbinfo_entries = []
        observations = env.reset()
        while episode_idx < 55:
            # observations = env.reset()

            scene = env.sim.semantic_annotations()
            depth_sensor_state = env.sim.get_agent_state().sensor_states["depth"]
            depth_hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV

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
            

            rgb_file = dataset_folder + f"/rgb{episode_idx:06d}"
            depth_file = dataset_folder + f"/depth{episode_idx:06d}"
            save_rgb(observations["rgb"], rgb_file)
            # save_rgb(observations["depth"], depth_file)
            bbox_file = dataset_folder + f"/bbs{episode_idx:06d}"
            save_bboxes(filtered_objects,
                        observations["rgb"],
                        depth_hfov,
                        depth_sensor_state, 
                        bbox_file)
            pc_file = dataset_folder + f"/pointcloud{episode_idx:06d}"
            save_pc(depth_hfov,
                    observations["depth"],
                    pc_file)

            info_entry, dbinfo_entries_sublist = gen_entries(depth_sensor_state,
                                                            episode_idx, 
                                                            len(dbinfo_entries),
                                                            rgb_file, 
                                                            pc_file, 
                                                            observations["rgb"], 
                                                            filtered_objects, 
                                                            desired_object, 
                                                            depth_sensor_state, 
                                                            depth_hfov)
            info_entries.append(info_entry)            
            dbinfo_entries.extend(dbinfo_entries_sublist)
            episode_idx += 1
            print("Done")
        
        # Save to "infos" file.
        with open(dataset_folder + "/infos.pkl", "wb") as f:
            pickle.dump(info_entries, f)
        # print(info_entries)
        
        # Save to "dbinfos" file.
        with open(dataset_folder + "/dbinfos.pkl", "wb") as f:
            pickle.dump({ desired_object : dbinfo_entries }, f)
        # print(dbinfo_entries)
        print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SECOND-usable dataset.")
    parser.add_argument('--dataset_folder', default="dataset_second", help="Dataset folder")
    parser.add_argument('--object', default="chair", help="Matterport object name")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder, args.object)
