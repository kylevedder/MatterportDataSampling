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

    def _save_boxes(self, split_name, names, bboxes, idx, save_vis):
        mesh = o3d.geometry.TriangleMesh()
        boxes = []
        for name, bb in zip(names, bboxes):
            box, img_bb, center, l, w, h, yaw = bb
            mesh += box
            boxes.append(np.array([name, *img_bb, *center, l, w, h, yaw]))

        boxes = np.array(boxes)
        p = Path(self.root_path + "/" + split_name + "/boxes/")
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{idx:06d}.txt").open('wb') as f:
            joblib.dump(boxes, f)
        if save_vis:
            o3d.io.write_triangle_mesh(str(p / f"{idx:06d}.ply"), mesh)
        return boxes

    def _save_pc(self, split_name, pc, idx, save_vis):
        p = Path(self.root_path + "/" + split_name + "/pc/")
        p.mkdir(parents=True, exist_ok=True)

        # Save with X, Y, Z, Intensity info.
        with (p / f"{idx:06d}.bin").open("wb") as bin_f:
            joblib.dump(pc, bin_f)

        if save_vis:
            # Save with X, Y, Z info only.
            PyntCloud(pd.DataFrame(data=pc,
                columns=["x", "y", "z"]))\
                .to_file(str(p / f"{idx:06d}.ply"))

    def _save_rgb(self, split_name, rgb_obs, idx, save_vis):
        if not save_vis:
            return
        p = Path(self.root_path + "/" + split_name + "/img/")
        p.mkdir(parents=True, exist_ok=True)
        plt.imshow(rgb_obs)
        plt.savefig(p / f"img{idx:06d}.png")
        plt.clf()

    def save_instance(self, split_name, idx, names, bboxes, pc, rgb_obs,
                      T_camera_world, T_image_camera, save_vis):
        assert T_image_camera.shape == (4, 4)
        assert T_camera_world.shape == (4, 4)
        self._save_boxes(split_name, names, bboxes, idx, save_vis)
        self._save_pc(split_name, pc, idx, save_vis)
        self._save_rgb(split_name, rgb_obs, idx, save_vis)
