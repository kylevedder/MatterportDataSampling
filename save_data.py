import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud
from pathlib import Path
import joblib
import os
import matplotlib.pyplot as plt


class SaveData:
    def __init__(self, root_path):
        self.root_path = root_path

    def _save_boxes(self, names, bboxes, idx, save_vis=False):
        mesh = o3d.geometry.TriangleMesh()
        boxes = []
        for name, bb in zip(names, bboxes):
            box, img_bb, center, l, w, h, yaw = bb
            mesh += box
            boxes.append(np.array([name, *img_bb, *center, l, w, h, yaw]))

        boxes = np.array(boxes)
        p = Path(self.root_path + "/boxes/")
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{idx:06d}.txt").open('wb') as f:
            joblib.dump(boxes, f)
        if save_vis:
            o3d.io.write_triangle_mesh(str(p / f"{idx:06d}.ply"), mesh)
        return boxes

    def _save_pc(self, pc, idx, save_vis=False):
        p = Path(self.root_path + "/pc/")
        p.mkdir(parents=True, exist_ok=True)

        # Save with X, Y, Z, Intensity info.
        with (p / f"{idx:06d}.bin").open("wb") as bin_f:
            print(pc.shape)
            joblib.dump(pc, bin_f)

        if save_vis:
            # Save with X, Y, Z info only.
            PyntCloud(pd.DataFrame(data=pc,
                columns=["x", "y", "z"]))\
                .to_file(str(p / f"{idx:06d}.ply"))

    def _save_transforms(self, idx, T_robot_camera, T_camera_robot,
                         T_camera_world, T_image_camera):
        p = Path(self.root_path + "/transforms/")
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"T_robot_camera{idx:06d}.bin").open("wb") as bin_f:
            joblib.dump(T_robot_camera, bin_f)
        with (p / f"T_camera_robot{idx:06d}.bin").open("wb") as bin_f:
            joblib.dump(T_camera_robot, bin_f)
        with (p / f"T_camera_world{idx:06d}.bin").open("wb") as bin_f:
            joblib.dump(T_camera_world, bin_f)
        with (p / f"T_image_camera{idx:06d}.bin").open("wb") as bin_f:
            joblib.dump(T_image_camera, bin_f)

    def save_instance(self,
                      idx,
                      names,
                      bboxes,
                      pc,
                      T_camera_world,
                      T_image_camera,
                      save_vis=False):
        assert T_image_camera.shape == (4, 4)
        assert T_camera_world.shape == (4, 4)
        boxes = self._save_boxes(names, bboxes, idx, save_vis)
        self._save_pc(pc, idx, save_vis)
        return boxes.shape[0]
