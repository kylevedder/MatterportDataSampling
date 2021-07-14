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

def save_bboxes(objs, sensor_state, filename):
    mesh = o3d.geometry.TriangleMesh()
        
    for obj in objs:
        bb = obj.obb
        box = o3d.geometry.TriangleMesh.create_box(*bb.half_extents)
        r1, r2, r3, r4 = bb.rotation
        box.rotate(Rotation.from_quat([r1, r2, r3, r4]).as_matrix(), 
                box.get_center())
        xt, yt, zt = bb.center
        box.translate((xt, yt, zt), relative=False)
        # cyl.translate(-sensor_state.position, relative=True)
        mesh += box
    
    # Construct transform from camera frame to world frame using camera position info.
    T_world_camera = np.eye(4)
    T_world_camera[0:3,0:3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    T_world_camera[0:3,3] = sensor_state.position
    # Invert to get transform from world frame to camera frame.
    T_camera_world = np.linalg.inv(T_world_camera)
    mesh.transform(T_camera_world)
    o3d.io.write_triangle_mesh(filename+".ply", mesh)

def save_pc(config, depth_obs, filename):
    # From https://aihabitat.org/docs/habitat-api/view-transform-warp.html

    # Assumes width and height are same for simplicity.
    W, H, d = depth_obs.shape
    assert(d == 1)
    depth_obs = depth_obs.reshape(W, H)
    assert(W == H)
    hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.0

    # Camera matrix constructed using pinhole camera model and assumption that W==H.
    K = np.array([
        [1 / np.tan(hfov / 2.), 0.0,                   0.0, 0.0],
        [0.0,                   1 / np.tan(hfov / 2.), 0.0, 0.0],
        [0.0,                   0.0,                   1.0, 0.0],
        [0.0,                   0.0,                   0.0, 1.0]
    ])
    K_inv = np.linalg.inv(K) # Converts image space coords to 3D camera frame coords.

    # Sample uniformly points in grid.
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up.
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    depth = depth_obs.reshape(1, W, H)
    xs = xs.reshape(1, W, H)
    ys = ys.reshape(1, W, H)
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)  # Flatten to 4 by (W*H) matrix.
    # Project to camera frame 3D XYZ from 2.5D pixel X, Pixel Y, Depth
    xy_c0 = np.matmul(K_inv, xys)
    PyntCloud(pd.DataFrame(data=xy_c0[:3].T,
        columns=["x", "y", "z"])).to_file(filename + ".ply")

    # Save in KITTI PC format of X, Y, Z, Intensity, with Intensity always 1.
    xy_c0[3] = 1
    with open(filename + ".bin", "wb") as bin_f:
        for e in xy_c0.T:
            bin_f.write(e.ravel().astype(np.float32))

    
    
def gen_entries(idx, rgb_file, pc_file, rgb_observations, objects, desired_object):
    info_entry = {
    "image" : {
        "image_idx" : idx,
        "image_path" : rgb_file + ".png",
        "image_shape" : np.array(list(rgb_observations.shape)[:2])
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
        "bbox" : np.zeros((len(objects), 4)), # IDK what this is
        "dimensions" : np.array([o.obb.half_extents for o in objects]),
        "location" : np.array([o.obb.center for o in objects]),
        "rotation_y" : np.zeros(len(objects)),
        "score" : np.zeros(len(objects)),
        "index" : np.array(range(len(objects))),
        "group_ids" : np.array(range(len(objects))),
        "difficulty" : np.zeros(len(objects)),
        "num_points_in_gt" : np.zeros(len(objects))
    }
    }
    extract_y = lambda rot: Rotation.from_quat(rot).as_euler('yxz', degrees=True)[:1]
    dbinfo_obj_entries = [{
            "name" : desired_object,
            "path" : pc_file + ".bin",
            "image_idx" : idx,
            "gt_idx" : 0, # IDK what this is
            "box3d_lidar" : np.concatenate([o.obb.center, o.obb.half_extents, extract_y(o.obb.rotation)]),
            "num_points_in_gt": 0, 
            "difficulty": 0, 
            "group_id": 0
    } for o in objects]
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
        while episode_idx < 10:
            observations = env.reset()

            scene = env.sim.semantic_annotations()

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
                continue

            print(f"Episode {episode_idx} has {len(filtered_objects)} {desired_object}")
            

            rgb_file = dataset_folder + f"/rgb{episode_idx:06d}"
            save_rgb(observations["rgb"], rgb_file)
            bbox_file = dataset_folder + f"/bbs{episode_idx:06d}"
            save_bboxes(filtered_objects,
                        env.sim.get_agent_state().sensor_states["depth"], 
                        bbox_file)
            pc_file = dataset_folder + f"/pointcloud{episode_idx:06d}"
            save_pc(config,
                    observations["depth"],
                    pc_file)

            info_entry, dbinfo_entries_sublist = gen_entries(episode_idx, 
                                                       rgb_file, 
                                                       pc_file, 
                                                       observations["rgb"], 
                                                       filtered_objects, 
                                                       desired_object)
            info_entries.append(info_entry)            
            dbinfo_entries.extend(dbinfo_entries_sublist)
            episode_idx += 1
        
        # Save to "infos" file.
        with open(dataset_folder + "/infos.pkl", "wb") as f:
            pickle.dump(info_entries, f)
        
        # Save to "dbinfos" file.
        with open(dataset_folder + "/dbinfos.pkl", "wb") as f:
            pickle.dump({ desired_object : dbinfo_entries }, f)
        print(dbinfo_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SECOND-usable dataset.")
    parser.add_argument('--dataset_folder', default="dataset", help="Dataset folder")
    parser.add_argument('--object', default="chair", help="Matterport object name")
    args = parser.parse_args()
    Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)
    main(args.dataset_folder, args.object)
