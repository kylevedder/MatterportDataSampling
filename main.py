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


def save_bboxes(objs, sensor_state, filename):
    mesh = o3d.geometry.TriangleMesh()

    DESIRED_OBJ = "chair"
    print(set([obj.category.name() for obj in objs]))
    objs = [obj for obj in objs if obj.category.name() == DESIRED_OBJ]
    print(f"{filename} has {len(objs)} {DESIRED_OBJ}")
        
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
        columns=["x", "y", "z"])).to_file(filename)



def main():
    config=habitat.get_config("task_mp3d.yaml")
    with habitat.Env(
        config=config
    ) as env:


        print("Environment creation successful")
        print("Agent acting inside environment.")
        for episode_idx in range(10):
            observations = env.reset()

            scene = env.sim.semantic_annotations()

            # Each id in the semantic image is an instance ID, which means we can lookup 
            # the actual *object* from the scene.
            # https://aihabitat.org/docs/habitat-sim/habitat_sim.scene.SemanticObject.html
            # Idea from https://github.com/facebookresearch/habitat-sim/issues/263#issuecomment-537295069            
            instance_id_to_obj = {int(obj.id.split("_")[-1]): obj for obj in scene.objects}
            to_catagory_id = np.vectorize(lambda e: instance_id_to_obj[e].category.index())
            # Extract the unique objects, get their Object Bounding Boxes.
            # https://aihabitat.org/docs/habitat-sim/habitat_sim.geo.OBB.html
            distinct_objects = [instance_id_to_obj[instance_id] for instance_id in set(observations["semantic"].flatten())]
            
            save_bboxes(distinct_objects,
                        env.sim.get_agent_state().sensor_states["depth"], 
                        "bbs{}".format(episode_idx))
            save_pc(config, 
                    observations["depth"], 
                    "pointcloud{}.ply".format(episode_idx))

            plt.imshow(observations["rgb"])
            plt.savefig("episode{}rgb.png".format(episode_idx))
            plt.clf()
            plt.imshow(to_catagory_id(observations["semantic"]))
            plt.colorbar()
            plt.savefig("episode{}semantic.png".format(episode_idx))
            plt.clf()
            plt.imshow(observations["depth"])
            plt.colorbar()
            plt.savefig("episode{}depth.png".format(episode_idx))
            plt.clf()            


if __name__ == "__main__":
    main()