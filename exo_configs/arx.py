"""ARX L5 robot with adhesive-mounted ArUco markers configuration."""
import numpy as np
from .exoskeleton import ExoskeletonConfig, LinkConfig, BLENDER_STL_DIR
from cv2 import aruco

ARX_MODEL_DIR = "robot_models/arx_l5"
SO100_MODEL_DIR = "robot_models/so100_model"
BOARD_IMG_DIR = "exo_hardware_files/board_imgs"

# Make aruco boards with correct ids
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board_ids_dict = { 'larger_base': np.arange(136, 172), 'link1': np.arange(73, 82), 'link3': np.arange(82, 91), 'link4': np.arange(91, 100), 'link6': np.arange(100, 109), }
link_boards = {}
# Create 3x3 boards for regular links
n_aruco_row, n_aruco_col = 3, 3
x = 4.5 / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
for name in ['link1', 'link3', 'link4', 'link6']:
    link_boards[name] = aruco.GridBoard( size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_dict[name].astype(np.int32).reshape(-1, 1))
# Create larger 6x6 board for base
n_aruco_row, n_aruco_col = 6, 6
x = (4.75 * 2) / ((n_aruco_row + (n_aruco_row - 1) / 10) * 100)
link_boards["larger_base"] = aruco.GridBoard( size=(n_aruco_row, n_aruco_col), markerLength=x, markerSeparation=x/10, dictionary=aruco_dict, ids=board_ids_dict['larger_base'].astype(np.int32).reshape(-1, 1))

class ARXConfig(ExoskeletonConfig):
    """Complete configuration for ARX L5 robot with adhesive ArUco mounts."""
    
    name = "ARX_L5"
    base_xml_path = f"{ARX_MODEL_DIR}/arx_l5.xml"
    background_xml_path = f"{SO100_MODEL_DIR}/background.xml"
    compiler_meshdir = f"{ARX_MODEL_DIR}/assets/"  # Set meshdir for robot meshes

    aruco_board_objects = link_boards
    
    # ArUco board patterns (image files)
    aruco_boards = { k: f"{BOARD_IMG_DIR}/{v}" for k, v in 
        {"larger_base": "larger_base.png", "shoulder": "shoulder.png", "upper_base": "upper_base.png",
         "lower_base": "lower_base.png", "roll": "roll.png"}.items() }
    
    links = {
        "larger_base": LinkConfig( mujoco_name="base_link", pybullet_name="base_link", robot_mesh_path=f"base_link.obj", exo_mesh_path=f"{BLENDER_STL_DIR}/arx_adhesive_base.stl",
                                    aruco_offset_pos=np.array([83.96,- 37.79,3.06]), aruco_offset_rot=np.array([0, 0, 0]), aruco_board_name="larger_base", board_length=0.095,),
        "link1": LinkConfig( mujoco_name="link1", pybullet_name="link1", robot_mesh_path=f"link1.obj", exo_mesh_path=f"{BLENDER_STL_DIR}/arx_adhesive_link1_cut.stl",
                                    aruco_offset_pos=np.array([15.56,-44.48,39.72]), aruco_offset_rot=np.array([np.pi/2, 0, 0]), aruco_board_name="shoulder", board_length=0.045,),
        "link3": LinkConfig( mujoco_name="link3", pybullet_name="link3", robot_mesh_path=f"link3.obj", exo_mesh_path=f"{BLENDER_STL_DIR}/arx_adhesive_link3.stl",
                                    aruco_offset_pos=np.array([-12.41,45.15,-65.67]), aruco_offset_rot=np.array([np.pi/2,0,np.pi]), aruco_board_name="upper_base", board_length=0.045,),
        "link4": LinkConfig( mujoco_name="link4", pybullet_name="link4", robot_mesh_path=f"link4.obj", exo_mesh_path=f"{BLENDER_STL_DIR}/arx_adhesive_link4.stl",
                                    aruco_offset_pos=np.array([3.28,42.56,-13.3]), aruco_offset_rot=np.array([np.pi/2, 0, np.pi]), aruco_board_name="lower_base", board_length=0.045,),
        "link6": LinkConfig( mujoco_name="link6", pybullet_name="link6", robot_mesh_path=f"link6.obj", exo_mesh_path=f"{BLENDER_STL_DIR}/arx_adhesive_link6.stl",
                                    aruco_offset_pos=np.array([45.48,-92.84,0]), aruco_offset_rot=np.array([np.pi/2, 0, 0]), aruco_board_name="roll", board_length=0.045,),
    }

# Convenience access
ARX_CONFIG = ARXConfig()
CONFIG = ARX_CONFIG  # Backwards compatibility