import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

obj_to_id = lambda obj: int(obj.id.split("_")[-1])


def _points_to_box(points):
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
    corner_distances = np.linalg.norm(other_corners -
                                      np.array([closest_x_point] * 3),
                                      2,
                                      axis=1)
    short_side_corner = other_corners[np.argmin(corner_distances)]

    short_side_len = np.linalg.norm(closest_x_point - short_side_corner, 2)
    short_side_center = (short_side_corner -
                         closest_x_point) / 2 + closest_x_point
    long_side_len = np.linalg.norm(center - short_side_center, 2) * 2

    delta_x, delta_y = (center - short_side_center)
    yaw = np.rad2deg(np.arctan2(delta_y, delta_x))

    return center3d, long_side_len, short_side_len, height, yaw


def _is_valid_box(obj,
                  box,
                  depth_hfov,
                  center,
                  l,
                  w,
                  h,
                  yaw,
                  img_bb,
                  img_sem,
                  img_pc,
                  min_pts_per_object,
                  max_distance=7,
                  min_sem_patch=0.15,
                  min_sem_points_in_box=0.8):
    
    # Reject Zs too high or too low
    x, y, z = center
    if z < -1 or z > 1:
        # print("outside Z", z)
        print("Reject: Outside Z")
        return False

    # Reject centers too far away
    xynorm = np.linalg.norm(center[:2], 2)
    if xynorm > max_distance:
        print("Reject: distance too far")
        # print("object center dist", xynorm, "outside max dist", max_distance)
        return False

    # Reject boxes whose center is outside of the view angle
    # Uses the classic crossproduct sign trick to check side of viewing line
    hov_ang = np.deg2rad(depth_hfov / 2)
    if np.cross([np.sin(hov_ang), np.cos(hov_ang), 0], [x, np.abs(y), 0])[2] >= 0:
        print("Reject: center outside ")
        return False
 
    # Compute points associated with that object
    vec3d_object_points = o3d.utility.Vector3dVector(img_pc[img_sem == obj_to_id(obj)])

    o3d_box = box.get_oriented_bounding_box()
    vec3d_inside_pts = o3d_box.get_point_indices_within_bounding_box(
        vec3d_object_points)

    # If fewer than min number of points inside object, reject
    if len(vec3d_inside_pts) < min_pts_per_object:
        print("Reject: too few points in box")
        return False

    # If too many points associated with the box are outside it, reject
    percent_points_in_box = len(vec3d_inside_pts) / len(vec3d_object_points)
    if percent_points_in_box < min_sem_points_in_box:
        print("Reject: too many outside box")
        return False
    return True


def _robot_corners_to_image_bb(robot_corners, W, H, T_image_robot):
    assert robot_corners.shape == (8, 3)
    robot_corners = np.concatenate([robot_corners, np.ones((8, 1))], axis=1)
    c = T_image_robot @ robot_corners.T

    def corners_to_pixel_pos(points, W, H):
        scale_arr = np.array([[-W / 2, 0], [0, H / 2]])
        translate_arr = np.array([[W / 2, H / 2]] * 8).T
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


def _make_bbox(obj, img_sem, img_pc, min_points_per_obj, depth_hfov,
               T_robot_world, T_image_robot):
    bb = obj.obb
    W, H = img_sem.shape
    box = o3d.geometry.TriangleMesh.create_box(*(bb.half_extents * 2))
    box.rotate(
        Rotation.from_quat(list(bb.rotation)).as_matrix(), box.get_center())
    box.translate(tuple(bb.center), relative=False)
    box.transform(T_robot_world)
    center, l, w, h, yaw = _points_to_box(np.asarray(box.vertices))
    box = o3d.geometry.TriangleMesh.create_box(l, w, h)
    box.rotate(
        Rotation.from_euler('z', yaw, degrees=True).as_matrix(),
        box.get_center())
    box.translate(tuple(center), relative=False)
    img_bb = _robot_corners_to_image_bb(np.asarray(box.vertices), W, H,
                                        T_image_robot)
    if not _is_valid_box(obj, box, depth_hfov, center, l, w, h, yaw, img_bb,
                         img_sem, img_pc, min_points_per_obj):
        return None
    return box, img_bb, center, l, w, h, yaw


def make_bboxes(semantic_obs,
                image_pc,
                scene,
                desired_objects,
                depth_hfov,
                T_robot_world,
                T_image_robot,
                min_points_per_obj=35):
    assert semantic_obs.shape == image_pc.shape[:2]
    if type(desired_objects) is not list:
        desired_objects = [desired_objects]
    # Each id in the semantic image is an instance ID, which means we can lookup
    # the actual *object* from the scene.
    # https://aihabitat.org/docs/habitat-sim/habitat_sim.scene.SemanticObject.html
    # Idea from https://github.com/facebookresearch/habitat-sim/issues/263#issuecomment-537295069
    instance_id_to_obj = {obj_to_id(obj): obj for obj in scene.objects}
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
    filtered_objects = [(obj,
                         _make_bbox(obj, semantic_obs, image_pc,
                                    min_points_per_obj, depth_hfov,
                                    T_robot_world, T_image_robot))
                        for obj in filtered_objects]
    filtered_objects = [(obj.category.name(), bbox)
                        for obj, bbox in filtered_objects if bbox is not None]
    if len(filtered_objects) <= 0:
        return [], []
    names, bboxes = zip(*filtered_objects)
    return names, bboxes