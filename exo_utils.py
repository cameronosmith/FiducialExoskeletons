"""Utilities for exoskeleton-based robot state estimation."""
import cv2
import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, Optional
from cv2 import aruco
import matplotlib.pyplot as plt

# ArUco dictionary
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

def do_est_aruco_pose(frame, aruco_dict, board, board_length, cameraMatrix=None, distCoeffs=None, pose_vis=None, corners_est=None,corners_vis=None):
    """Estimate ArUco pose - copied from aruco_helpers.py"""
    from cv2 import aruco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    if corners_est is None: corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco.DetectorParameters())
    else: corners, ids, rejected,  = corners_est
    if ids is None: return -1
    
    corners_vis = cv2.aruco.drawDetectedMarkers(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if corners_vis is None else corners_vis, corners, ids)
    W, H = frame.shape[:2][::-1]
    f0 = max(W, H)//2
    if distCoeffs is None: distCoeffs = dist_coeffs_init = np.zeros((8, 1), dtype=np.float64)
    if cameraMatrix is None:
        camera_matrix_init = np.array([[f0, 0, W/2.0], [0, f0, H/2.0], [0, 0, 1.0]], dtype=np.float64)
        flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | 
                 cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_ASPECT_RATIO | 
                 cv2.CALIB_FIX_PRINCIPAL_POINT)
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco( corners, ids, np.array([len(ids)]), board, gray.shape[::-1], camera_matrix_init, dist_coeffs_init, flags=flags)

    # Use solvePnPGeneric to remove pose ambiguity
    obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(board, corners, ids)
    if obj_pts is None or obj_pts.size == 0: return -1
    obj_pts = obj_pts.reshape(-1, 3).astype(np.float32)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    
    # Request all valid PnP solutions
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE)
    
    # Choose the pose whose board normal points toward the camera
    best_idx = 0
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R_mat, _ = cv2.Rodrigues(rv)
        normal = R_mat @ np.array([0., 0., 1.])  # board +Z in camera coords
        board_in_front = tv[2] > 0  # Z must be positive
        faces_camera = normal[2] < 0  # normal points toward camera
        if board_in_front and faces_camera:
            best_idx = i
            break
    rvec, tvec = rvecs[best_idx][:, 0], tvecs[best_idx][:, 0]

    # Compute object points in camera frame for confidence estimate
    obj_pts_cam = (cv2.Rodrigues(rvec)[0] @ obj_pts.T + tvec.reshape(3, 1)).T
    img_reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cameraMatrix, distCoeffs)
    img_reproj = img_reproj.reshape(-1, 2)

    R_mat = cv2.Rodrigues(rvec)[0]
    center_offset_board = np.array([board_length/2, board_length/2, 0], dtype=np.float64)
    tvec = tvec + R_mat.dot(center_offset_board)
    
    # Draw axes - use fixed small size (0.03m = 30mm)
    pose_vis = cv2.drawFrameAxes(corners_vis, cameraMatrix, np.zeros((1, 5)), rvec, tvec, 0.03)
    
    est_aruco_pose = np.eye(4)
    est_aruco_pose[:3, 3] = tvec
    est_aruco_pose[:3, :3] = R_mat
    est_aruco_pose[:, 1:-1] *= -1

    return { "est_aruco_pose": est_aruco_pose, "pose_vis": pose_vis, "corners_vis": corners_vis, "corners": (ids.flatten(), corners), "cameraMatrix": cameraMatrix, "rtvec": (rvec, tvec), "distCoeffs": distCoeffs, "corners_est": (corners, ids, rejected), "obj_img_pts":(obj_pts_cam,img_pts),}


def get_link_poses_from_robot( robot_config, model: mujoco.MjModel, data: mujoco.MjData):
    """Extract link poses from robot's current state.
    """
    link_poses = {}
    for link_name, cfg in robot_config.links.items():
        # Get the robot body (not the exoskeleton mocap body)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.pybullet_name)
        
        # Validate quaternion
        quat_wxyz = data.xquat[body_id]
        if np.linalg.norm(quat_wxyz) < 0.01: continue
        
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = data.xpos[body_id]
        link_poses[link_name] = pose
    
    return link_poses


def position_exoskeleton_meshes( robot_config, model: mujoco.MjModel, data: mujoco.MjData, link_poses: Dict[str, np.ndarray] = None):
    """Position virtual exoskeleton meshes """
    for link_name, cfg in robot_config.links.items():
        # Get link pose
        if link_poses is not None and link_name in link_poses:
            # Use detected pose (already in world coordinates from detection)
            link_pose = link_poses[link_name]
            link_pos = link_pose[:3, 3]
            
            # Validate rotation matrix
            rot_mat = link_pose[:3, :3]
            if not np.allclose(np.linalg.det(rot_mat), 1.0, atol=0.01): continue
            
            link_rot = R.from_matrix(rot_mat)
            link_quat_wxyz = link_rot.as_quat()[[3, 0, 1, 2]]  # xyzw to wxyz
        else:
            # Use robot's current state 
            body_id = model.body(cfg.pybullet_name).id
            link_pos = data.xpos[body_id]
            link_quat_wxyz = data.xquat[body_id]
            try: link_rot = R.from_quat(link_quat_wxyz[[1, 2, 3, 0]])  # wxyz to xyzw
            except: continue

        # Set link mesh position and orientation
        link_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{link_name}_link_mesh")]
        data.mocap_pos[link_mocap_id] = link_pos
        data.mocap_quat[link_mocap_id] = link_quat_wxyz

        # Set exo mesh (same as link)
        exo_mesh_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{link_name}_exo_mesh")]
        data.mocap_pos[exo_mesh_mocap_id] = link_pos
        data.mocap_quat[exo_mesh_mocap_id] = link_quat_wxyz
        
        # Set aruco plane with offset
        plane_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{link_name}_exo_plane")
        # Apply offset in link frame
        data.mocap_pos[model.body_mocapid[plane_body_id]] = link_pos + link_rot.apply(cfg.aruco_offset_pos / 1000)
        data.mocap_quat[model.body_mocapid[plane_body_id]] = (link_rot * R.from_euler('xyz', cfg.aruco_offset_rot)).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)

#Detect link poses from ArUco markers in image.
def detect_link_poses( rgb: np.ndarray, robot_config, visualize: bool = False, cam_K: np.ndarray = None):
    # Convert to uint8 if needed
    if rgb.dtype == np.float32 or rgb.dtype == np.float64: rgb = (rgb * 255).astype(np.uint8)
    
    # Detect all markers once
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners_cache = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=cv2.aruco.DetectorParameters())
    vis_img = rgb.copy()
    
    # Draw all detected markers
    corners, ids, rejected = corners_cache
    cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)

    # Detect poses for all links (FIRST PASS: all in camera frame)
    link_poses = {}
    obj_img_pts = {}
    for link_name in robot_config.links.keys():
        cfg = robot_config.links[link_name]
        
        result = do_est_aruco_pose( rgb, ARUCO_DICT, robot_config.aruco_board_objects[link_name], cfg.board_length, cameraMatrix=cam_K, corners_est=corners_cache, corners_vis=vis_img)
        if result == -1: continue
        vis_img = result['pose_vis']
        
        # Compute link pose in camera frame
        link_to_aruco = np.block([ [R.from_euler('xyz', cfg.aruco_offset_rot).as_matrix(), cfg.aruco_offset_pos[:, None] / 1000], [np.zeros((1, 3)), 1] ])
        link_poses[link_name] = result['est_aruco_pose'] @ np.linalg.inv(link_to_aruco)
        obj_img_pts[link_name] = result['obj_img_pts']
        if link_name == "larger_base": cam_K,cam_pose,link_poses["larger_base"] = result['cameraMatrix'],link_poses[link_name],np.eye(4)
        else: link_poses[link_name] = np.linalg.inv(cam_pose) @ link_poses[link_name]
    
    if visualize: plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB));plt.show()
    return link_poses, cam_pose, cam_K, corners_cache,vis_img,obj_img_pts


def detect_and_set_link_poses(rgb: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData, robot_config, visualize: bool = False, cam_K: np.ndarray = None):
    """Detect link poses from ArUco markers in image and set virtual exoskeleton meshes to match."""
    link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_link_poses(rgb, robot_config, visualize=visualize, cam_K=cam_K)
    position_exoskeleton_meshes(robot_config, model, data, link_poses)
    return link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts


def render_from_camera_pose(model: mujoco.MjModel, data: mujoco.MjData, cam_pose: np.ndarray, cam_K: np.ndarray, height: int, width: int,segmentation: bool = False):
    """Render scene from estimated camera pose.
    
    Returns:
        Rendered image (numpy array)
    """
    from mujoco.renderer import Renderer
    renderer = Renderer(model, height=height, width=width)
    cam_id = model.cam('estimated_camera').id

    if segmentation: renderer.enable_segmentation_rendering()
    
    # Set camera position and orientation
    data.cam_xpos[cam_id] = np.linalg.inv(cam_pose)[:3, 3]
    data.cam_xmat[cam_id] = (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ cam_pose[:3, :3]).T.reshape(-1)
    
    # Set FOV from camera matrix
    model.cam_fovy[cam_id] = np.degrees(2 * np.arctan(height / (2 * cam_K[1, 1])))
    
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()


def estimate_robot_state( model: mujoco.MjModel, data: mujoco.MjData, robot_config, link_poses, ik_iterations: int = 15, position_cost: float = 1.0, orientation_cost: float = 0.03):
    """Estimate robot joint configuration from RGB image using ArUco markers.
    """
    # Solve IK to find joint configuration
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)  # Initialize with current robot state
    
    tasks = []
    # Make tasks for each detected link
    link_name_map = {cfg.pybullet_name: link_name for link_name, cfg in robot_config.links.items()}
    for link_name in link_poses:
        if link_name not in link_name_map.values(): continue
        pyb_name = [k for k, v in link_name_map.items() if v == link_name][0]
        task = mink.FrameTask( frame_name=pyb_name, frame_type="body", position_cost=position_cost, orientation_cost=orientation_cost)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{link_name}_link_mesh")
        task.set_target(mink.SE3(wxyz_xyz=np.concatenate([data.xquat[body_id], data.xpos[body_id]])))
        tasks.append(task)
    
    # Solve IK
    for _ in range(ik_iterations):
        vel = mink.solve_ik( configuration, tasks, 0.005, "daqp", limits=[mink.ConfigurationLimit(model=model)])
        configuration.integrate_inplace(vel, 0.005)
    
    return configuration

