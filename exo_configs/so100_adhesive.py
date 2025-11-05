"""SO100 robot with adhesive-mounted ArUco markers configuration."""
import numpy as np
import cv2
from cv2 import aruco
from .exoskeleton import ExoskeletonConfig, LinkConfig, BLENDER_STL_DIR

SO100_MODEL_DIR = "robot_models/so100_model"
BOARD_IMG_DIR = "exo_hardware_files/board_imgs"

# Create ArUco boards with correct IDs from aruco_helpers.py
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Make aruco boards with correct ids
board_ids_dict = { 'larger_base': np.arange(136, 172), 'shoulder': np.arange(73, 82), 'upper_base': np.arange(82, 91), 'lower_base': np.arange(91, 100), 'roll': np.arange(100, 109), 'fixed_gripper': np.arange(109, 118), 'moveable_gripper': np.arange(118, 127), }
link_boards = {}
# Create 3x3 boards for regular links
n_aruco_row, n_aruco_col = 3, 3
x = 4.5 / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
for name in ['shoulder', 'upper_base', 'lower_base', 'roll', 'fixed_gripper', 'moveable_gripper']:
    link_boards[name] = aruco.GridBoard( size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_dict[name].astype(np.int32).reshape(-1, 1))
# Create larger 6x6 board for base
n_aruco_row, n_aruco_col = 6, 6
x = (4.75 * 2) / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
link_boards["larger_base"] = aruco.GridBoard( size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_dict['larger_base'].astype(np.int32).reshape(-1, 1))

class SO100AdhesiveConfig(ExoskeletonConfig):
    """Complete configuration for SO100 robot with adhesive ArUco mounts."""
    
    name = "SO100_Adhesive"
    base_xml_path = f"{SO100_MODEL_DIR}/so_arm100.xml"
    background_xml_path = f"{SO100_MODEL_DIR}/background.xml"
    compiler_meshdir = f"{SO100_MODEL_DIR}/assets/"  # Set meshdir for robot meshes
    
    # ArUco board patterns (image files)
    aruco_boards = { k: f"{BOARD_IMG_DIR}/{v}" for k, v in 
        {"larger_base":  "larger_base.png", "shoulder": "shoulder.png", "upper_base": "upper_base.png",
        "lower_base":   "lower_base.png", "roll": "roll.png", "fixed_gripper": "fixed_gripper.png",}.items() }
    
    # ArUco board objects (for detection)
    aruco_board_objects = link_boards


    links = {
        "larger_base": LinkConfig( mujoco_name="larger_base", pybullet_name="Base", robot_mesh_path=f"Base.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/adhesive_base.stl",
                                    aruco_offset_pos=np.array([-107, -30.5, 4]), aruco_offset_rot=np.array([0, 0, np.pi/2]), aruco_board_name="larger_base", board_length=0.095,),
        "shoulder": LinkConfig( mujoco_name="shoulder", pybullet_name="Rotation_Pitch", robot_mesh_path=f"Rotation_Pitch.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/adhesive_shoulder.stl",
                                    aruco_offset_pos=np.array([0, 11.44, 77]), aruco_offset_rot=np.array([-np.pi/2, 0, 0]), aruco_board_name="shoulder", board_length=0.045,),
        "upper_base": LinkConfig( mujoco_name="upper_base", pybullet_name="Upper_Arm", robot_mesh_path=f"Upper_Arm.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/adhesive_upper_arm.stl", aruco_offset_pos=np.array([-38.91, 16.20, -12.00]),
                                    aruco_offset_rot=np.array([0, -np.pi/2, 0]), aruco_board_name="upper_base", board_length=0.045,),
        "lower_base": LinkConfig( mujoco_name="lower_base", pybullet_name="Lower_Arm", robot_mesh_path=f"Lower_Arm.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/adhesive_lower_arm_exo_redo3.stl",
                                    aruco_offset_pos=np.array([-36.6, 12.02, 11.71]), aruco_offset_rot=np.array([0, -np.pi/2, 0]), aruco_board_name="lower_base", board_length=0.045,),
        "roll": LinkConfig( mujoco_name="roll", pybullet_name="Wrist_Pitch_Roll", robot_mesh_path=f"Wrist_Pitch_Roll.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/roll_redo_both_shift.stl",
                                    aruco_offset_pos=np.array([-37., -37.89, 6.88]), aruco_offset_rot=np.array([0, -np.pi/2, 0]), aruco_board_name="roll", board_length=0.045,),
        "fixed_gripper": LinkConfig( mujoco_name="fixed_gripper", pybullet_name="Fixed_Jaw", robot_mesh_path=f"Fixed_Jaw.stl", exo_mesh_path=f"{BLENDER_STL_DIR}/fixed_gripper_exo.stl",
                                    aruco_offset_pos=np.array([35.7, -30.9, 0]), aruco_offset_rot=np.array([0, np.pi/2, 0]), aruco_board_name="fixed_gripper", board_length=0.045,),
    }

# Convenience access
SO100_ADHESIVE_CONFIG = SO100AdhesiveConfig()