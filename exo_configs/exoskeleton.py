"""Base exoskeleton configuration framework."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

BLENDER_STL_DIR = "../../../exo_hardware_files"
SO100_MODEL_DIR = "robot_models/so100_model"
BOARD_IMG_DIR = "exo_hardware_files/board_imgs"

@dataclass
class LinkConfig:
    """Configuration for a single robot link with ArUco marker."""
    mujoco_name: str          # Name in MuJoCo XML
    pybullet_name: str        # Name for PyBullet/IK solver
    robot_mesh_path: str      # Path to robot's visual mesh
    exo_mesh_path: str        # Path to exoskeleton/mount mesh
    aruco_offset_pos: np.ndarray  # [x, y, z] offset from link to ArUco (mm)
    aruco_offset_rot: np.ndarray  # [rx, ry, rz] euler angles (radians)
    aruco_board_name: str     # Name of ArUco board pattern
    board_length: float       # Physical size of board (meters)


class ExoskeletonConfig:
    """Base configuration class for robots with ArUco marker exoskeletons."""
    
    # Override these in subclass
    name: str = "Robot"
    base_xml_path: str = ""
    background_xml_path: str = "robot_models/so100_model/background.xml"
    compiler_meshdir: str = "./"
    
    # ArUco board patterns (override in subclass)
    aruco_boards: Dict[str, str] = {}
    
    # Physical board sizes
    board_length_small: float = 0.045  # 3x3 markers
    board_length_large: float = 0.095  # 6x6 markers
    
    # Link configurations (override in subclass)
    links: Dict[str, LinkConfig] = {}

    exo_alpha: float = 0.6
    aruco_alpha: float = 1.0  # Transparency for ArUco planes (0.0 = invisible, 1.0 = opaque)
    
    def __init__(self):
        """Initialize and generate XML."""
        self.xml = self._generate_xml()
    
    def _generate_xml(self) -> str:
        """Generate complete MuJoCo XML with exoskeletons and ArUco planes."""
        
        # Header
        xml = f"""<mujoco>
                    <compiler angle="radian" meshdir="{self.compiler_meshdir}"/>
                    <include file="{self.base_xml_path}"/>
                    <include file="{self.background_xml_path}"/>
                    <visual> <global offheight="4100" offwidth="4100"/> </visual>
                    <asset>"""
        
        # Robot and exoskeleton meshes
        for name, cfg in self.links.items():
            # Robot meshes from SO100 model are already in correct units (meters), no scaling needed
            # Exoskeleton meshes from Blender need .001 scale (mm to m conversion)
            exo_scale = ' scale=".001 .001 .001"' if cfg.exo_mesh_path.endswith('.stl') else ''
            xml += f"""
                    <mesh name="{name}_link_stl" file="{cfg.robot_mesh_path}" inertia="shell"/>
                    <mesh name="{name}_exo_stl" file="{cfg.exo_mesh_path}"{exo_scale} inertia="shell"/>"""
        
        # ArUco plane meshes
        xml += f"""<mesh name="aruco_plane_small" file="{BLENDER_STL_DIR}/plane.obj" scale="0.15 0.15 .001" inertia="shell"/>
                   <mesh name="aruco_plane_large" file="{BLENDER_STL_DIR}/plane.obj" scale="0.3166666 0.3166666 .001" inertia="shell"/>"""
        
        # Textures and materials
        for board_name, img_path in self.aruco_boards.items():
            xml += f"""
                <texture name="{board_name}_tex" type="2d" file="{img_path}"/>
                <material name="{board_name}_mat" texture="{board_name}_tex" rgba="1 1 1 1"/>"""
        xml += """ </asset>
                <worldbody>
                    <camera name="estimated_camera" pos="0 0 0" quat="1 0 0 0" fovy="95"/>
                """
        
        # Mocap bodies for each link
        for name, cfg in self.links.items():
            plane_size = "large" if cfg.board_length > 0.05 else "small"
            xml += f"""
                    <!-- {name} -->
                    <body mocap="true" name="{name}_link_mesh">
                    <geom type="mesh" mesh="{name}_link_stl" contype="0" conaffinity="0" rgba="1 0 0 {self.exo_alpha}" />
                    </body>
                    <body mocap="true" name="{name}_exo_mesh">
                    <geom type="mesh" mesh="{name}_exo_stl" contype="0" conaffinity="0" rgba="0 1 0 {self.exo_alpha}" />
                    </body>
                    <body mocap="true" name="{name}_exo_plane">
                    <geom type="mesh" mesh="aruco_plane_{plane_size}" contype="0" conaffinity="0" material="{cfg.aruco_board_name}_mat" rgba="1 1 1 {self.aruco_alpha}" />
                    </body>
                """
        xml += """  </worldbody> </mujoco>"""
        
        return xml