import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud

class SaveKitti:
  def __init__(self, root_path):
    self.root_path = root_path
    self.P0 = np.zeros((3, 4)).reshape((12,))
    self.P1 = np.zeros((3, 4)).reshape((12,))
    self.P3 = np.zeros((3, 4)).reshape((12,))
    self.R0_rect = np.eye(3).reshape((9,))
    self.Tr_imu_to_velo = np.zeros((3, 4)).reshape((12,))


  def save_calibration(self, idx):
    name_datas = [("P0", self.P0), 
                  ("P1", self.P1), 
                  ("P2", self.P2), 
                  ("P3", self.P3), 
                  ("R0_rect", self.R0_rect), 
                  ("Tr_velo_to_cam", self.Tr_velo_to_cam), 
                  ("Tr_imu_to_velo", self.Tr_imu_to_velo)]
    with open(self.root_path + f"/calib/{idx:06d}.txt") as f:
      for name, data in name_datas:
        f.write(name + ": " + ' '.join(data) + '\n')      

  
  def save_rgb(self, rgb, idx):
    pass


  def save_boxes(self, names, bboxes, image_bboxes, T_second_world, idx):
    mesh = o3d.geometry.TriangleMesh()
    with open(self.root_path + f"/label_2/{idx:06d}.txt") as f:
      for name, bb, img_bb in zip(names, bboxes, image_bboxes):
        # name # idx 0
        truncation = 0 # idx 1
        occlusion = 0 # idx 2
        left, top, right, bottom = img_bb # idx 4, 5, 6, 7
        l, w, h = bb.half_extents # idx 8, 9, 10
        x, y, z = bb.center # 11, 12, 13
        rotation_y = Rotation.from_quat(*bb.rotation).as_euler('y', degrees=False) # idx 14
        alpha = rotation_y # This is wrong, should fix. idx 3
        score = 1 # idx 15
        data = [name, truncation, occlusion, alpha, left, top, right, bottom, h, w, l, x, y, z, rotation_y, score]
        f.write(' '.join(data) + "\n")

        # Save .ply for visualization
        box = o3d.geometry.TriangleMesh.create_box(*(bb.half_extents * 2))
        r1, r2, r3, r4 = bb.rotation
        box.rotate(Rotation.from_quat([r1, r2, r3, r4]).as_matrix(), 
                box.get_center())
        xt, yt, zt = bb.center
        box.translate((xt, yt, zt), relative=False)
        mesh += box        

    # Put mesh in camera frame
    mesh.transform(T_second_world)
    o3d.io.write_triangle_mesh(self.root_path + f"/label_2/{idx:06d}.ply", mesh)
        

  def save_pc(self, pc, idx):
    PyntCloud(pd.DataFrame(data=pc[:3].T,
        columns=["x", "y", "z"]))\
        .to_file(self.root_path +f"/velodyne/{idx:06d}.ply")
    # Save in KITTI PC binary format of X, Y, Z, Intensity, with Intensity always 1.
    pc[3] = 1
    with open(self.root_path +f"/velodyne/{idx:06d}.bin", "wb") as bin_f:
        bin_f.write(pc.T.ravel().astype(np.float32))


  def save_instance(self, idx, rgb, names, bboxes, image_bboxes, pc, T_second_camera, T_camera_world, T_camera_img):
    assert T_camera_img.shape == (4, 4)
    assert T_camera_world.shape == (4, 4)
    self.Tr_velo_to_cam = T_camera_world[:3].reshape((12,))
    self.P2 = T_camera_img.T[:3].reshape((12,))


    self.save_calibration(idx)
    self.save_rgb(rgb, idx)
    self.save_boxes(names, bboxes, image_bboxes, idx)
    self.save_pc(pc, idx)


