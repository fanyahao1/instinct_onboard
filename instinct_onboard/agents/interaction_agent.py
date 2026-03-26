"""
InteractionAgent for instinct_onboard.

This agent combines:
- Depth perception from ParkourAgent (0-depth_encoder.onnx)
- Motion reference tracking from ShadowingAgent (0-motion_ref.onnx, forward_kinematics.onnx)
- Object state observations (object_pos, object_ori, wrist_object_contact)

Reference:
- ParkourAgent: depth image processing and velocity command
- ShadowingAgent: motion reference encoding and FK
- TrackerAgent: NPZ file loading pattern
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort
import prettytable
import quaternion
import ros2_numpy as rnp
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2, PointField

from instinct_onboard.agents.base import ColdStartAgent, OnboardAgent
from instinct_onboard.normalizer import Normalizer
from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.utils import (
    CircularBuffer,
    inv_quat,
    quat_rotate_inverse,
    quat_slerp_batch,
    quat_to_tan_norm_batch,
    yaw_quat,
)


@dataclass
class MotionData:
    """Motion data structure for interaction task.
    
    Compared to TrackerAgent's MotionData, this adds object_data field
    for object state trajectories.
    """
    framerate: float
    # Joint orders must match the robot joint names in simulation order.
    joint_pos: np.ndarray  # (N, num_joints)
    joint_vel: np.ndarray  # (N, num_joints)
    base_pos: np.ndarray   # (N, 3)
    base_quat: np.ndarray   # (N, 4)
    total_num_frames: int
    # Object data - stored as dict for flexibility
    # Expected keys: "box" -> {pos, quat, lin_vel, ang_vel, contact, validity}
    object_data: Optional[Dict[str, np.ndarray]] = None


def load_motion_data(
    motion_file: str,
    robot_joint_names: list[str],
    target_framerate: float,
) -> MotionData:
    """Load motion data from NPZ file, compatible with interaction task.
    
    Handles both standard motion data and motion data with object trajectories.
    """
    motion_data = np.load(motion_file, allow_pickle=True)
    framerate = motion_data["framerate"].item()

    motion_joint_names_all = motion_data["joint_names"].tolist()
    motion_joint_to_robot_joint_ids = [motion_joint_names_all.index(j_name) for j_name in robot_joint_names]

    joint_pos = motion_data["joint_pos"][:, motion_joint_to_robot_joint_ids]
    joint_pos_ = np.concatenate([joint_pos[0:1], joint_pos])
    joint_vel = (joint_pos_[1:] - joint_pos_[:-1]) * framerate
    base_pos = motion_data["base_pos_w"]
    base_quat = motion_data["base_quat_w"]
    total_num_frames = motion_data["joint_pos"].shape[0]

    # Try to load object data if available
    object_data = None
    if "object_data" in motion_data.files:
        object_data = motion_data["object_data"].item()  # Convert from numpy dict to python dict

    motion = MotionData(
        framerate=framerate,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pos=base_pos,
        base_quat=base_quat,
        total_num_frames=total_num_frames,
        object_data=object_data,
    )

    return match_framerate(motion, target_framerate)


def match_framerate(motion_data: MotionData, target_framerate: float) -> MotionData:
    """Resample motion data to target framerate."""
    if motion_data.framerate == target_framerate:
        return motion_data

    motion_length_s = motion_data.total_num_frames / motion_data.framerate
    new_total_num_frames = math.floor(motion_length_s * target_framerate)
    new_frame_idxs = np.linspace(0, motion_data.total_num_frames - 1, new_total_num_frames)
    floor = np.floor(new_frame_idxs).astype(int)
    ceil = np.ceil(new_frame_idxs).astype(int)
    frac = new_frame_idxs - floor

    new_joint_pos = (1 - frac)[:, None] * motion_data.joint_pos[floor] + frac[:, None] * motion_data.joint_pos[ceil]
    joint_pos_ = np.concatenate([new_joint_pos[0:1], new_joint_pos])
    new_joint_vel = (joint_pos_[1:] - joint_pos_[:-1]) * target_framerate
    new_base_pos = (1 - frac)[:, None] * motion_data.base_pos[floor] + frac[:, None] * motion_data.base_pos[ceil]
    new_base_quat = quat_slerp_batch(motion_data.base_quat[floor], motion_data.base_quat[ceil], frac)

    # Resample object data if available
    new_object_data = None
    if motion_data.object_data is not None:
        new_object_data = {}
        for obj_name, obj_arrays in motion_data.object_data.items():
            if isinstance(obj_arrays, dict):
                # Object state dict with pos, quat, lin_vel, ang_vel, contact, validity
                new_object_data[obj_name] = {
                    key: (1 - frac)[:, None] * val[floor] + frac[:, None] * val[ceil]
                    if val.ndim == 2 and val.shape[0] == motion_data.total_num_frames
                    else val
                    for key, val in obj_arrays.items()
                }
            else:
                # Simple array
                new_object_data[obj_name] = (
                    (1 - frac)[:, None] * obj_arrays[floor] + frac[:, None] * obj_arrays[ceil]
                )

    return MotionData(
        framerate=target_framerate,
        joint_pos=new_joint_pos.astype(np.float32),
        joint_vel=new_joint_vel.astype(np.float32),
        base_pos=new_base_pos.astype(np.float32),
        base_quat=new_base_quat.astype(np.float32),
        total_num_frames=new_total_num_frames,
        object_data=new_object_data,
    )


class InteractionAgent(OnboardAgent):
    """Agent for interaction tasks combining depth perception, motion reference, and object state.
    
    This agent loads:
    - actor.onnx: Policy network
    - 0-depth_encoder.onnx: Depth image encoder
    - 0-motion_ref.onnx: Motion reference encoder (optional)
    - forward_kinematics.onnx: FK for link positions
    
    Observations include:
    - Proprioception: joint_pos, joint_vel, projected_gravity, base_ang_vel, last_action
    - Depth: depth_latent (from depth encoder)
    - Motion reference: joint_pos_ref, joint_vel_ref, position_ref, rotation_ref
    - Object state: object_pos, object_ori (relative to base frame)
    - Contact: wrist_object_contact
    
    Since motion data is not yet available, object-related observations return zeros.
    """

    rs_resolution = (480, 270)
    rs_frequency = 60

    def __init__(
        self,
        logdir: str,
        motion_file_dir: str,
        ros_node: RealNode,
        depth_vis: bool = True,
        pointcloud_vis: bool = True,
        target_motion_framerate: float = 50.0,
        object_name: str = "box",
    ):
        super().__init__(logdir, ros_node)
        self.target_motion_framerate = target_motion_framerate
        self.object_name = object_name
        self.ort_sessions: dict[str, ort.InferenceSession] = dict()
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()
        self._load_all_motions(motion_file_dir)
        
        # Depth visualization
        self.depth_vis = depth_vis
        if self.depth_vis:
            self.debug_depth_publisher = ros_node.create_publisher(Image, "/debug/depth_image", 10)
        else:
            self.debug_depth_publisher = None
        
        self.pointcloud_vis = pointcloud_vis
        if self.pointcloud_vis:
            self.debug_pointcloud_publisher = ros_node.create_publisher(PointCloud2, "/debug/pointcloud", 10)
        else:
            self.debug_pointcloud_publisher = None

        # Link pose cache for FK
        self.link_pos_: Optional[np.ndarray] = None
        self.link_quat_: Optional[np.ndarray] = None
        self.link_tannorm_: Optional[np.ndarray] = None

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        
        all_obs_names = list(self.obs_funcs.keys())
        
        # Separate depth observations
        self.depth_obs_names = [obs_name for obs_name in all_obs_names if "depth" in obs_name]
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if "depth" not in obs_name]
        
        # Check for motion reference observations
        self.motion_ref_obs_names = []
        self.proprio_obs_names_no_motion_ref = []
        if "encoder_configs" in self.agent_cfg.get("policy", {}):
            encoder_cfg = self.agent_cfg["policy"]["encoder_configs"]
            if "motion_ref" in encoder_cfg:
                self.motion_ref_obs_names = encoder_cfg["motion_ref"].get("component_names", [])
        
        # Proprioception excludes motion_ref if using encoder
        if self.motion_ref_obs_names:
            self.proprio_obs_names_no_motion_ref = [
                n for n in self.proprio_obs_names if n not in self.motion_ref_obs_names
            ]
        else:
            self.proprio_obs_names_no_motion_ref = self.proprio_obs_names
        
        print(f"InteractionAgent proprioception names (no motion ref): {self.proprio_obs_names_no_motion_ref}")
        print(f"InteractionAgent depth observation names: {self.depth_obs_names}")
        print(f"InteractionAgent motion reference names: {self.motion_ref_obs_names}")
        
        table = prettytable.PrettyTable()
        table.field_names = ["Observation Name", "Function"]
        for obs_name, func in self.obs_funcs.items():
            table.add_row([obs_name, func.__name__])
        print("Observation functions:")
        print(table)
        
        # Parse depth config if depth observations exist
        if self.depth_obs_names:
            self._parse_depth_image_config()

    def _parse_action_config(self):
        super()._parse_action_config()
        self._zero_action_joints = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        import re
        for action_names, action_config in self.cfg["actions"].items():
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if "default_joint_names" in action_config:
                    for _, joint_name_expr in enumerate(action_config["default_joint_names"]):
                        if re.search(joint_name_expr, name):
                            self._zero_action_joints[i] = 1.0

    def _parse_depth_image_config(self):
        """Parse depth image processing configuration from env.yaml."""
        self.output_resolution = [
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        ]
        self.depth_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["depth_range"]
        
        if self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["normalize"]:
            self.depth_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["output_range"]
        else:
            self.depth_output_range = self.depth_range
        
        if "crop_and_resize" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.crop_region = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"]["crop_region"]
        if "gaussian_blur" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.gaussian_kernel_size = (
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
            )
            self.gaussian_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["sigma"]
        if "blind_spot" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.blind_spot_crop = self.cfg["scene"]["camera"]["noise_pipeline"]["blind_spot"]["crop_region"]
        
        self.depth_width = (
            self.output_resolution[0] - self.crop_region[2] - self.crop_region[3]
            if hasattr(self, "crop_region")
            else self.output_resolution[0]
        )
        self.depth_height = (
            self.output_resolution[1] - self.crop_region[0] - self.crop_region[1]
            if hasattr(self, "crop_region")
            else self.output_resolution[1]
        )
        
        # For sample resize
        square_size = int(self.rs_resolution[0] // self.output_resolution[0])
        rows, cols = self.rs_resolution[1], self.rs_resolution[0]
        center_y_coords = np.arange(self.output_resolution[1]) * square_size + square_size // 2
        center_x_coords = np.arange(self.output_resolution[0]) * square_size + square_size // 2
        y_grid, x_grid = np.meshgrid(center_y_coords, center_x_coords, indexing="ij")
        self.y_valid = np.clip(y_grid, 0, rows - 1)
        self.x_valid = np.clip(x_grid, 0, cols - 1)
        
        # For downsample history
        if "history_skip_frames" in self.cfg["observations"]["policy"]["depth_image"]["params"]:
            downsample_factor = self.cfg["observations"]["policy"]["depth_image"]["params"]["history_skip_frames"]
        else:
            downsample_factor = self.cfg["observations"]["policy"]["depth_image"]["params"]["time_downsample_factor"]
        
        frames = int(
            (self.cfg["scene"]["camera"]["data_histories"]["distance_to_image_plane_noised"] - 1) / downsample_factor
            + 1
        )
        sim_frequency = int(1 / self.cfg["scene"]["camera"]["update_period"])
        real_downsample_factor = int(self.rs_frequency / sim_frequency * downsample_factor)
        self.depth_obs_indices = np.linspace(-1 - real_downsample_factor * (frames - 1), -1, frames).astype(int)
        print(f"Depth observation downsample indices: {self.depth_obs_indices}")
        self.depth_image_buffer = CircularBuffer(length=self.rs_frequency)

    def _parse_observation_function(self, obs_name: str, obs_config: dict):
        """Override to handle depth_image function name mapping."""
        obs_func = obs_config["func"].split(":")[-1]
        if obs_func == "depth_image":
            obs_name = "depth_latent"
            if hasattr(self, f"_get_{obs_name}_obs"):
                self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_name}_obs")
                return
            else:
                raise ValueError(f"Unknown observation function for observation {obs_name}")
        return super()._parse_observation_function(obs_name, obs_config)

    def _load_models(self):
        """Load ONNX models for the agent."""
        ort_execution_providers = ort.get_available_providers()
        
        # Load actor model
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        
        # Load depth encoder if exists
        depth_encoder_path = os.path.join(self.logdir, "exported", "0-depth_encoder.onnx")
        if os.path.exists(depth_encoder_path):
            self.ort_sessions["depth_encoder"] = ort.InferenceSession(
                depth_encoder_path, providers=ort_execution_providers
            )
            self.has_depth_encoder = True
        else:
            self.has_depth_encoder = False
            print("Warning: No depth encoder found, depth perception disabled")
        
        # Load motion reference encoder if exists
        motion_ref_path = os.path.join(self.logdir, "exported", "0-motion_ref.onnx")
        if os.path.exists(motion_ref_path):
            self.ort_sessions["motion_ref"] = ort.InferenceSession(
                motion_ref_path, providers=ort_execution_providers
            )
            self.has_motion_ref_encoder = True
        else:
            self.has_motion_ref_encoder = False
            print("Warning: No motion reference encoder found")
        
        # Load forward kinematics if exists
        fk_path = os.path.join(self.logdir, "exported", "forward_kinematics.onnx")
        if os.path.exists(fk_path):
            self.ort_sessions["fk"] = ort.InferenceSession(fk_path, providers=ort_execution_providers)
            self.has_fk = True
        else:
            self.has_fk = False
            print("Warning: No forward kinematics model found")
        
        # Load normalizer if exists
        normalizer_path = os.path.join(self.logdir, "exported", "policy_normalizer.npz")
        if os.path.exists(normalizer_path):
            self.normalizer = Normalizer(load_path=normalizer_path)
        else:
            self.normalizer = None
        
        print(f"Loaded ONNX models from {self.logdir}")

    def _load_all_motions(self, motion_file_dir: str):
        """Load all motion files from directory."""
        self.all_motion_datas: dict[str, MotionData] = dict()
        
        if not os.path.exists(motion_file_dir):
            print(f"Warning: Motion directory {motion_file_dir} does not exist, no motions loaded")
            self.all_motion_datas = {}
            self.motion_data = None
            return
        
        for motion_file in os.listdir(motion_file_dir):
            if not motion_file.endswith(".npz"):
                continue
            try:
                motion = load_motion_data(
                    os.path.join(motion_file_dir, motion_file),
                    self.ros_node.sim_joint_names,
                    self.target_motion_framerate,
                )
                self.all_motion_datas[motion_file] = motion
            except Exception as e:
                print(f"Warning: Failed to load motion {motion_file}: {e}")
        
        if self.all_motion_datas:
            self.motion_data = list(self.all_motion_datas.values())[0]
            self.ros_node.get_logger().info(
                f"Loaded {len(self.all_motion_datas)} motions from {motion_file_dir}"
            )
        else:
            self.motion_data = None
            self.ros_node.get_logger().warn(f"No valid motions found in {motion_file_dir}")
        
        # Prepare frame indices for motion reference
        if self.motion_data is not None and "scene" in self.cfg and "motion_reference" in self.cfg["scene"]:
            self.motion_num_frames = self.cfg["scene"]["motion_reference"]["num_frames"]
            self.motion_frame_indices_offset = np.arange(self.motion_num_frames).astype(float)
            if self.cfg["scene"]["motion_reference"]["data_start_from"] == "one_frame_interval":
                self.motion_frame_indices_offset += 1
            self.motion_frame_indices_offset *= (
                self.cfg["scene"]["motion_reference"]["frame_interval_s"] * self.target_motion_framerate
            )
            self.motion_frame_indices_offset = self.motion_frame_indices_offset.astype(int)
        else:
            self.motion_num_frames = 0
            self.motion_frame_indices_offset = np.array([0])
        
        self.motion_cursor_idx = 0

    def _update_links_poses(self):
        """Update current link positions using forward kinematics."""
        if not self.has_fk:
            return
        
        joint_pos = self.ros_node.joint_pos_
        fk_input_name = self.ort_sessions["fk"].get_inputs()[0].name
        output = self.ort_sessions["fk"].run(None, {fk_input_name: joint_pos[None, :]})
        self.link_pos_ = output[0][0]    # (num_links, 3)
        self.link_quat_ = output[1][0]   # (num_links, 4)
        self.link_tannorm_ = quat_to_tan_norm_batch(self.link_quat_)

    def refresh_depth_frame(self):
        """Refresh and preprocess depth frame from RealSense."""
        self.ros_node.refresh_rs_data()
        depth_image_np: np.ndarray = self.ros_node.rs_depth_data
        depth_image = cv2.resize(depth_image_np, self.output_resolution, interpolation=cv2.INTER_NEAREST)
        
        if hasattr(self, "crop_region"):
            shape = depth_image.shape
            x1, x2, y1, y2 = self.crop_region
            depth_image = depth_image[x1 : shape[0] - x2, y1 : shape[1] - y2]
        
        # Inpaint small holes
        mask = (depth_image < 0.2).astype(np.uint8)
        depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)
        
        # Apply blind spot masking
        if hasattr(self, "blind_spot_crop"):
            shape = depth_image.shape
            x1, x2, y1, y2 = self.blind_spot_crop
            depth_image[:x1, :] = 0
            depth_image[shape[0] - x2 :, :] = 0
            depth_image[:, :y1] = 0
            depth_image[:, shape[1] - y2 :] = 0
        
        # Gaussian blur
        if hasattr(self, "gaussian_kernel_size"):
            depth_image = cv2.GaussianBlur(
                depth_image, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma
            )
        
        # Normalize
        filt_m = np.clip(depth_image, self.depth_range[0], self.depth_range[1])
        filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        output_norm = filt_norm * (
            self.depth_output_range[1] - self.depth_output_range[0]
        ) + self.depth_output_range[0]
        
        self.depth_image_buffer.append(output_norm)

    def _get_depth_image_downsample_obs(self) -> np.ndarray:
        """Get downsampled depth image observation."""
        self.refresh_depth_frame()
        return self.depth_image_buffer.buffer[self.depth_obs_indices, ...]

    def _get_depth_latent_obs(self) -> np.ndarray:
        """Get depth latent embedding from encoder."""
        if not self.has_depth_encoder:
            # Return zeros if no depth encoder
            return np.zeros((1, 128), dtype=np.float32)  # Assuming 128-dim embedding
        
        depth_obs = (
            self._get_depth_image_downsample_obs()
            .reshape(1, -1, self.depth_height, self.depth_width)
            .astype(np.float32)
        )
        return self.ort_sessions["depth_encoder"].run(
            None, {self.ort_sessions["depth_encoder"].get_inputs()[0].name: depth_obs}
        )[0]

    def _get_object_pos_obs(self) -> np.ndarray:
        """Get object position relative to robot base.
        
        Returns:
            np.ndarray: Object position in base frame, shape (3,)
        
        Note: Currently returns zeros since motion data with object trajectories
        is not yet available. Will be updated when data is provided.
        """
        if self.motion_data is None or self.motion_data.object_data is None:
            return np.zeros(3, dtype=np.float32)
        
        obj_data = self.motion_data.object_data.get(self.object_name)
        if obj_data is None:
            return np.zeros(3, dtype=np.float32)
        
        # Get object position at current cursor
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        pos_b = obj_data.get("pos", np.zeros((self.motion_data.total_num_frames, 3)))
        
        # Transform to base frame (simplified - just return relative position)
        robot_pos = self.ros_node.joint_pos_  # This is joint pos, not base pos
        # For now, just return the object position in world frame transformed by gravity
        # A proper implementation would use base position from motion data
        return pos_b[cursor].astype(np.float32)

    def _get_object_ori_obs(self) -> np.ndarray:
        """Get object orientation in tangent-normal form.
        
        Returns:
            np.ndarray: Object orientation, shape (6,)
        
        Note: Currently returns zeros since motion data with object trajectories
        is not yet available.
        """
        if self.motion_data is None or self.motion_data.object_data is None:
            return np.zeros(6, dtype=np.float32)
        
        obj_data = self.motion_data.object_data.get(self.object_name)
        if obj_data is None:
            return np.zeros(6, dtype=np.float32)
        
        # Get object orientation at current cursor
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        quat = obj_data.get("quat", np.zeros((self.motion_data.total_num_frames, 4)))
        
        # Convert to tangent-normal
        quat_array = quaternion.from_float_array(quat[cursor:cursor+1])
        tannorm = quat_to_tan_norm_batch(quaternion.as_float_array(quat_array))
        
        return tannorm.flatten().astype(np.float32)

    def _get_wrist_object_contact_obs(self) -> np.ndarray:
        """Get wrist-object contact forces.
        
        Returns:
            np.ndarray: Contact forces [left, right], shape (2,)
        
        Note: Currently returns zeros since contact sensor data is not yet
        available via ROS. Will be updated when contact sensors are integrated.
        """
        return np.zeros(2, dtype=np.float32)

    def _get_link_pos_b_obs(self) -> np.ndarray:
        """Get current link positions from FK.
        
        Returns:
            np.ndarray: Link positions in base frame, shape (num_links, 3)
        """
        if self.link_pos_ is None:
            return np.zeros((1, 3), dtype=np.float32)
        return self.link_pos_.astype(np.float32)

    def _get_link_tannorm_b_obs(self) -> np.ndarray:
        """Get current link orientations in tangent-normal form.
        
        Returns:
            np.ndarray: Link orientations, shape (num_links, 6)
        """
        if self.link_tannorm_ is None:
            return np.zeros((1, 6), dtype=np.float32)
        return self.link_tannorm_.astype(np.float32)

    def _get_motion_ref_observation(self) -> np.ndarray:
        """Get encoded motion reference observation.
        
        Returns:
            np.ndarray: Motion reference embedding, shape (embedding_dim,)
        """
        if not self.has_motion_ref_encoder or self.motion_data is None:
            return np.zeros((1, 128), dtype=np.float32)
        
        # Pack motion reference observations
        motion_ref_obs = []
        for motion_ref_obs_name in self.motion_ref_obs_names:
            if hasattr(self, f"_get_{motion_ref_obs_name}_cmd_obs"):
                obs_fn = getattr(self, f"_get_{motion_ref_obs_name}_cmd_obs")
            elif hasattr(self, f"_get_{motion_ref_obs_name}_obs"):
                obs_fn = getattr(self, f"_get_{motion_ref_obs_name}_obs")
            else:
                continue
            obs_term_value = obs_fn()
            time_dim = obs_term_value.shape[0]
            motion_ref_obs.append(
                obs_term_value.reshape(1, time_dim, -1).astype(np.float32)
            )
        
        if not motion_ref_obs:
            return np.zeros((1, 128), dtype=np.float32)
        
        motion_ref_obs = np.concatenate(motion_ref_obs, axis=-1)
        
        # Run motion reference encoder
        motion_ref_input_name = self.ort_sessions["motion_ref"].get_inputs()[0].name
        motion_ref_output = self.ort_sessions["motion_ref"].run(
            None, {motion_ref_input_name: motion_ref_obs}
        )[0]
        
        return motion_ref_output

    def _get_joint_pos_ref_command_obs(self) -> np.ndarray:
        """Get reference joint positions at future frames.
        
        Returns:
            np.ndarray: Joint positions, shape (num_frames, num_joints)
        """
        if self.motion_data is None:
            return np.zeros((1, self.ros_node.NUM_JOINTS), dtype=np.float32)
        
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        frame_idx = cursor + self.motion_frame_indices_offset
        frame_idx = np.clip(frame_idx, 0, self.motion_data.total_num_frames - 1)
        
        ref_joint_pos = self.motion_data.joint_pos[frame_idx] - self.ros_node.default_joint_pos[None, :]
        return ref_joint_pos.astype(np.float32)

    def _get_joint_vel_ref_command_obs(self) -> np.ndarray:
        """Get reference joint velocities at future frames.
        
        Returns:
            np.ndarray: Joint velocities, shape (num_frames, num_joints)
        """
        if self.motion_data is None:
            return np.zeros((1, self.ros_node.NUM_JOINTS), dtype=np.float32)
        
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        frame_idx = cursor + self.motion_frame_indices_offset
        frame_idx = np.clip(frame_idx, 0, self.motion_data.total_num_frames - 1)
        
        return self.motion_data.joint_vel[frame_idx].astype(np.float32)

    def _get_position_ref_command_obs(self) -> np.ndarray:
        """Get reference base position at future frames.
        
        Returns:
            np.ndarray: Base position relative to current, shape (num_frames, 3)
        """
        if self.motion_data is None:
            return np.zeros((1, 3), dtype=np.float32)
        
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        frame_idx = cursor + self.motion_frame_indices_offset
        frame_idx = np.clip(frame_idx, 0, self.motion_data.total_num_frames - 1)
        
        # Return base position in base frame (simplified)
        return self.motion_data.base_pos[frame_idx].astype(np.float32)

    def _get_rotation_ref_command_obs(self) -> np.ndarray:
        """Get reference base rotation in tangent-normal form.
        
        Returns:
            np.ndarray: Base rotation, shape (num_frames, 6)
        """
        if self.motion_data is None:
            return np.zeros((1, 6), dtype=np.float32)
        
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        frame_idx = cursor + self.motion_frame_indices_offset
        frame_idx = np.clip(frame_idx, 0, self.motion_data.total_num_frames - 1)
        
        # Convert quaternions to tangent-normal
        base_quat = self.motion_data.base_quat[frame_idx]
        tannorm = quat_to_tan_norm_batch(base_quat)
        
        return tannorm.astype(np.float32)

    def _get_position_b_ref_command_obs(self) -> np.ndarray:
        """Alias for _get_position_ref_command_obs to match env.yaml command_name."""
        return self._get_position_ref_command_obs()

    # === Aliases for env.yaml naming ===
    def _get_object_position_obs(self) -> np.ndarray:
        """Alias for _get_object_pos_obs to match env.yaml."""
        return self._get_object_pos_obs()

    def _get_object_orientation_tannorm_obs(self) -> np.ndarray:
        """Alias for _get_object_ori_obs to match env.yaml."""
        return self._get_object_ori_obs()

    def _get_object_position_error_obs(self) -> np.ndarray:
        """Alias for _get_object_pos_obs (error version uses same raw value)."""
        return self._get_object_pos_obs()

    def _get_object_orientation_error_tannorm_obs(self) -> np.ndarray:
        """Alias for _get_object_ori_obs (error version uses same raw value)."""
        return self._get_object_ori_obs()

    # === Proxy methods to ros_node for proprioception ===
    def _get_projected_gravity_obs(self) -> np.ndarray:
        """Proxy to ros_node's projected_gravity from IMU."""
        return self.ros_node._get_projected_gravity_obs()

    def _get_base_ang_vel_obs(self) -> np.ndarray:
        """Proxy to ros_node's base angular velocity from IMU."""
        return self.ros_node._get_base_ang_vel_obs()

    def _get_joint_pos_rel_obs(self) -> np.ndarray:
        """Proxy to ros_node's joint position (relative version)."""
        return self.ros_node._get_joint_pos_obs()

    def _get_joint_vel_rel_obs(self) -> np.ndarray:
        """Proxy to ros_node's joint velocity (relative version)."""
        return self.ros_node._get_joint_vel_rel_obs()

    def _get_last_action_obs(self) -> np.ndarray:
        """Proxy to ros_node's last action."""
        return self.ros_node._get_last_action_obs()

    # === Object reference stubs (from motion dataset) ===
    def _get_object_reference_position_obs(self) -> np.ndarray:
        """Get object reference position from motion data (world frame, shape (3,))."""
        if self.motion_data is None or self.motion_data.object_data is None:
            return np.zeros(3, dtype=np.float32)
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        obj_pos = self.motion_data.object_data.get("box", {}).get("pos", np.zeros(3))
        if obj_pos.ndim == 2:
            obj_pos = obj_pos[cursor]
        return obj_pos.astype(np.float32)

    def _get_object_reference_orientation_tannorm_obs(self) -> np.ndarray:
        """Get object reference orientation from motion data (tan-norm, shape (6,))."""
        if self.motion_data is None or self.motion_data.object_data is None:
            return np.zeros(6, dtype=np.float32)
        cursor = min(self.motion_cursor_idx, self.motion_data.total_num_frames - 1)
        obj_quat = self.motion_data.object_data.get("box", {}).get("quat", np.zeros(4))
        if obj_quat.ndim == 2:
            obj_quat = obj_quat[cursor]
        tannorm = quat_to_tan_norm_batch(obj_quat[np.newaxis, :])[0]
        return tannorm.astype(np.float32)

    # === Contact sensor stub ===
    def _get_seat_object_contact_obs(self) -> np.ndarray:
        """Get contact between seat/buttocks and object (placeholder, returns zeros)."""
        return np.zeros(2, dtype=np.float32)

    def reset(self, motion_name: str = None):
        """Reset the agent state.
        
        Args:
            motion_name: Optional motion file name to load
        """
        super().reset()
        
        if motion_name is not None and motion_name in self.all_motion_datas:
            self.motion_data = self.all_motion_datas[motion_name]
        elif self.all_motion_datas:
            self.motion_data = list(self.all_motion_datas.values())[0]
        
        self.match_to_current_heading()
        self.motion_cursor_idx = 0
        
        if self.motion_data is not None:
            self.ros_node.get_logger().info(
                f"InteractionAgent reset with motion: {motion_name or 'default'}"
            )

    def get_done(self) -> bool:
        """Check if current motion is done.
        
        Returns:
            bool: True if motion cursor has reached the end
        """
        if self.motion_data is None:
            return True
        return (self.motion_cursor_idx + self.motion_frame_indices_offset[-1]) >= (
            self.motion_data.total_num_frames - 1
        )

    def match_to_current_heading(self):
        """Match the motion's 0-th frame to the current robot heading."""
        if self.motion_data is None:
            return
        
        root_quat_w = quaternion.from_float_array(self.ros_node._get_quat_w_obs())
        quat_w_ref = quaternion.from_float_array(self.motion_data.base_quat[0])
        quat_err = root_quat_w * inv_quat(quat_w_ref)
        heading_err_quat = yaw_quat(quat_err)
        heading_err_quat_ = np.stack(
            [heading_err_quat for _ in range(len(self.motion_data.base_quat))], axis=0
        )
        
        # Update base_quat_w for each frame
        motion_quats = quaternion.from_float_array(self.motion_data.base_quat)
        updated_quats = heading_err_quat_ * motion_quats
        self.motion_data.base_quat = quaternion.as_float_array(updated_quats)
        
        # Update base_pos_w for each frame
        current_pos_w = self.motion_data.base_pos[0]
        rel_pos = self.motion_data.base_pos - self.motion_data.base_pos[0:1]
        rotated_rel_pos = quaternion.rotate_vectors(heading_err_quat, rel_pos)
        self.motion_data.base_pos = rotated_rel_pos + current_pos_w[None, :]

    def step(self) -> tuple[np.ndarray, bool]:
        """Perform a single step of the agent.
        
        Returns:
            tuple: (action, done) where action is the policy output and
                   done indicates if the motion has completed
        """
        # Update motion cursor
        self.motion_cursor_idx += 1
        done = self.get_done()
        
        # Clamp cursor to valid range
        if self.motion_data is not None:
            self.motion_cursor_idx = min(
                self.motion_cursor_idx, self.motion_data.total_num_frames - 1
            )
        
        # Update link poses via FK
        self._update_links_poses()
        
        # Pack observations
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names_no_motion_ref:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(np.reshape(obs_term_value, (1, -1)).astype(np.float32))
        
        # Add depth embedding if available
        if self.has_depth_encoder and self.depth_obs_names:
            depth_latent = self._get_depth_latent_obs()
            proprio_obs.append(depth_latent.astype(np.float32))
        
        # Add motion reference embedding if available
        if self.has_motion_ref_encoder:
            motion_ref_emb = self._get_motion_ref_observation()
            proprio_obs.append(motion_ref_emb.astype(np.float32))
        
        # Concatenate all observations
        if proprio_obs:
            actor_input = np.concatenate(proprio_obs, axis=-1)
        else:
            actor_input = np.zeros((1, 128), dtype=np.float32)
        
        # Apply normalization if available
        if self.normalizer is not None:
            actor_input = self.normalizer.normalize(actor_input.flatten()).reshape(1, -1)
        
        # Run actor network
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: actor_input})[0]
        action = action.reshape(-1)
        
        # Reconstruct full action including zeroed joints
        mask = (self._zero_action_joints == 0).astype(bool)
        full_action = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        full_action[mask] = action
        
        # Publish debug info
        if self.debug_depth_publisher is not None:
            depth_obs = (
                self._get_depth_image_downsample_obs()
                .reshape(-1, self.depth_height, self.depth_width)
            )
            depth_image_msg_data = np.asanyarray(
                depth_obs[-1] * 255 * 2, dtype=np.uint16
            )
            depth_image_msg = rnp.msgify(Image, depth_image_msg_data, encoding="16UC1")
            depth_image_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            depth_image_msg.header.frame_id = "realsense_depth_link"
            self.debug_depth_publisher.publish(depth_image_msg)
        
        if self.debug_pointcloud_publisher is not None and hasattr(self, "depth_range"):
            depth_obs_raw = (
                self._get_depth_image_downsample_obs()
                .reshape(-1, self.depth_height, self.depth_width)[-1]
            )
            pointcloud_msg = self.ros_node.depth_image_to_pointcloud_msg(
                depth_obs_raw * self.depth_range[1] + self.depth_range[0]
            )
            self.debug_pointcloud_publisher.publish(pointcloud_msg)
        
        return full_action, done

    def get_cold_start_agent(
        self,
        startup_step_size: float = 0.2,
        kpkd_factor: float = 2.0,
    ) -> ColdStartAgent:
        """Get a cold start agent configured with this agent's settings.
        
        Args:
            startup_step_size: Step size for cold start
            kpkd_factor: Factor to multiply P/D gains
        
        Returns:
            ColdStartAgent configured for transition to this agent
        """
        return ColdStartAgent(
            startup_step_size=startup_step_size,
            ros_node=self.ros_node,
            joint_target_pos=self.default_joint_pos,
            action_scale=self.action_scale,
            action_offset=self.action_offset,
            p_gains=self.p_gains * kpkd_factor,
            d_gains=self.d_gains * kpkd_factor,
        )
