import numpy as np

def make_pc(depth_obs,
            T_robot_camera,
            height_max=2,
            height_min=-2,
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
    pc = pc[pc[:, 2] < height_max]
    pc = pc[pc[:, 2] > height_min]

    return pc, image_pc