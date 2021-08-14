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

  def _points_to_box(self, points):
    assert points.shape == (8, 3)
    height = points[:, 2].max() - points[:, 2].min()

    def extract_four_corners(points):
      duped_points = list(points[:, :2])
      safe_idx = 0
      while safe_idx < len(duped_points):
        safe_val = duped_points[safe_idx]
        for query_idx in range(safe_idx + 1, len(duped_points)):
          query_val = duped_points[query_idx]
          if np.linalg.norm(safe_val - query_val, 1) < 0.01:
            # Duplicate detected
            duped_points.pop(query_idx)
            break
        safe_idx += 1

      assert len(duped_points) == 4
      return np.array(duped_points)
    
    center3d = np.mean(points, axis=0)
    corners = extract_four_corners(points)
    center = np.mean(corners, axis=0)
    closest_x_point_idx = np.argmin(corners[:, 0])
    closest_x_point = corners[closest_x_point_idx]

    mask = np.ones(4, bool)
    mask[closest_x_point_idx] = False
    other_corners = corners[mask]
    corner_distances = np.linalg.norm(other_corners - np.array([closest_x_point] * 3), 2, axis=1)
    short_side_corner = other_corners[np.argmin(corner_distances)]

    short_side_len = np.linalg.norm(closest_x_point - short_side_corner, 2)
    short_side_center = (short_side_corner - closest_x_point) / 2 + closest_x_point
    long_side_len = np.linalg.norm(center - short_side_center, 2) * 2

    delta_x, delta_y = (center - short_side_center)
    yaw = np.rad2deg(np.arctan2(delta_y, delta_x))

    return center3d, long_side_len, short_side_len, height, yaw


  def _is_valid_box(self, center):
    if center[2] < -1 or center[2] > 1:
      print("outside Z", center[2])
      return False
    if np.linalg.norm(center[:2], 2) > 8:
      print("outside max range", np.linalg.norm(center[:2], 2))
      return False
    return True

  
  def _second_corners_to_image_bb(self, second_corners, rgb_img, T_image_second):
    W, H, _ = rgb_img.shape
    assert second_corners.shape == (8, 3)
    second_corners = np.concatenate([second_corners, np.ones((8, 1))], axis=1)
    c = T_image_second @ second_corners.T

    def corners_to_pixel_pos(points, W, H):
      scale_arr = np.array([[-W/2, 0],[0, H/2]])
      translate_arr = np.array([[W/2, H/2]]*8).T
      return (scale_arr @ points + translate_arr)

    def pixel_pos_to_bbox(points):
        max_x, min_x = np.max(points[0]), np.min(points[0])
        max_y, min_y = np.max(points[1]), np.min(points[1])
        # left, top, right, bottom format
        return np.array([min_x, min_y, max_x, max_y])

    c = c[:2] / c[2]
    c = corners_to_pixel_pos(c, W, H)
    c = pixel_pos_to_bbox(c)
    return c

    

  def _save_boxes(self, names, bboxes, rgb_img, T_second_world, T_image_second, idx):
    mesh = o3d.geometry.TriangleMesh()
    boxes = []
    for name, bb in zip(names, bboxes):
      # Bounding boxes are not stored in l,w,h format; they are stored in decending order
      # side lengths.
      # Use open3d to transform corners into 3D bounding box, then extract BEV-based info
      # for the bounding box
      box = o3d.geometry.TriangleMesh.create_box(*(bb.half_extents * 2))
      box.rotate(Rotation.from_quat(list(bb.rotation)).as_matrix(), box.get_center())
      box.translate(tuple(bb.center), relative=False)
      box.transform(T_second_world)
      center, l, w, h, yaw = self._points_to_box(np.asarray(box.vertices))
      if not self._is_valid_box(center):
        continue
      box = o3d.geometry.TriangleMesh.create_box(l, w, h)
      box.rotate(Rotation.from_euler('z', yaw, degrees=True).as_matrix(), box.get_center())
      box.translate(tuple(center), relative=False)
      img_bb = self._second_corners_to_image_bb(np.asarray(box.vertices), rgb_img, T_image_second)
      mesh += box
      boxes.append(np.array([name, *img_bb, *center, l, w, h, yaw]))
    
    boxes = np.array(boxes)
    if boxes.shape[0] > 0:
      p = Path(self.root_path + "/boxes/")
      p.mkdir(parents=True, exist_ok=True)
      with (p / f"{idx:06d}.txt").open('wb') as f:
        joblib.dump(boxes, f)
      o3d.io.write_triangle_mesh(str(p / f"{idx:06d}.ply"), mesh)
    return boxes
        

  def _save_pc(self, pc, T_second_camera, idx):
    pc = T_second_camera @ pc
    pc = pc.T.astype(np.float32)

    # Chop below height cutoff.
    kMaxHeightCutoff = 2
    pc = pc[pc[:,2] < kMaxHeightCutoff]

    p = Path(self.root_path + "/pc/")
    p.mkdir(parents=True, exist_ok=True)
    # Save with X, Y, Z info only.
    PyntCloud(pd.DataFrame(data=pc[:,:3],
        columns=["x", "y", "z"]))\
        .to_file(str(p / f"{idx:06d}.ply"))
    
    # Save with X, Y, Z, Intensity info.
    pc[:,3] = 1
    with (p / f"{idx:06d}.bin").open("wb") as bin_f:
      print(pc.shape)
      joblib.dump(pc, bin_f)

  def _save_transforms(self, idx, T_second_camera, T_camera_second, T_camera_world, T_image_camera):
    p = Path(self.root_path + "/transforms/")
    p.mkdir(parents=True, exist_ok=True)
    with (p / f"T_second_camera{idx:06d}.bin").open("wb") as bin_f:
      joblib.dump(T_second_camera, bin_f)
    with (p / f"T_camera_second{idx:06d}.bin").open("wb") as bin_f:
      joblib.dump(T_camera_second, bin_f)
    with (p / f"T_camera_world{idx:06d}.bin").open("wb") as bin_f:
      joblib.dump(T_camera_world, bin_f)
    with (p / f"T_image_camera{idx:06d}.bin").open("wb") as bin_f:
      joblib.dump(T_image_camera, bin_f)

  def save_instance(self, idx, rgb_img, names, bboxes, pc, T_second_camera, T_camera_second, T_camera_world, T_image_camera):
    assert T_image_camera.shape == (4, 4)
    assert T_camera_world.shape == (4, 4)
    boxes = self._save_boxes(names, 
                             bboxes, 
                             rgb_img, 
                             T_second_camera @ T_camera_world, 
                             T_image_camera @ T_camera_second, 
                             idx)
    if boxes.shape[0] > 0:
      self._save_pc(pc, T_second_camera, idx)
    return boxes.shape[0]


