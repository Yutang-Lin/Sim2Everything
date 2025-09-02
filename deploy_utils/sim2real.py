# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Yutang-Lin.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import mujoco
import mujoco.viewer
import numpy as np  
import torch
import time
import sys
import os
import time
from copy import deepcopy
from .math_utils import (
    euler_xyz_from_quat,
    quat_apply_inverse,
)
import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
from .crc import CRC
from .gamepad import Gamepad, parse_remote_data
from .base_env import BaseEnv

class UnitreeEnv(BaseEnv, Node):
    simulated = False

    def __init__(self, control_freq: int = 100, 
                 joint_order: list[str] | None = None,
                 action_joint_names: list[str] | None = None,
                 release_time_delta: float = 0.0,
                 align_time: bool = True,
                 align_step_size: float = 0.00005,
                 align_tolerance: float = 2.0,
                 init_rclpy: bool = True,
                 spin_timeout: float = 0.001,
                 simulated_state: bool = False,
                 **kwargs):
        """
        Initialize MuJoCo environment
        
        Args:
            control_freq: Control frequency in Hz (must be <= simulation_freq)
            joint_order: List of joint names specifying the order of joints for control and observation.
            action_joint_names: List of joint names that are actuated (subset of joint_order).
            release_time_delta: Delta time to step_complete return True before control dt reach
            align_time: Whether to adjust release_time_delta to align the control frequency with the real-time step frequency
            align_step_size: Step size to auto-adjust the release_time_delta
            init_rclpy: Whether to initialize rclpy
            spin_timeout: Timeout for rclpy.spin_once
        """
        BaseEnv.__init__(self, control_freq=control_freq,
                         joint_order=joint_order,
                         action_joint_names=action_joint_names,
                         release_time_delta=release_time_delta,
                         align_time=align_time,
                         align_step_size=align_step_size,
                         align_tolerance=align_tolerance,
                         init_rclpy=init_rclpy,
                         spin_timeout=spin_timeout,
                         simulated_state=simulated_state,
                         **kwargs) # check kwargs
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq
        self.release_time_delta = release_time_delta
        self.align_time = align_time
        self.align_step_size = align_step_size
        self.align_tolerance = align_tolerance
        self.spin_timeout = spin_timeout
        self.simulated_state = simulated_state
        assert self.joint_limits is not None, "joint_limits must be provided when using sim2real"
        if isinstance(self.joint_limits, (list, tuple)):
            self.joint_limits = torch.tensor(self.joint_limits)
        elif isinstance(self.joint_limits, np.ndarray):
            self.joint_limits = torch.from_numpy(self.joint_limits)
        assert isinstance(self.joint_limits, torch.Tensor)
        assert self.joint_limits.ndim == 2
        assert self.joint_limits.shape[1] == 2
        assert self.joint_limits.shape[0] == len(joint_order)
        self.ignore_limit_joints = self.emergency_stop_condition.get('ignore_limit_joints', [])

        # State variables
        self.step_count = 0

        assert joint_order is not None, "joint_order must be provided"
        
        # PD gains (can be modified) - now per-joint arrays
        self.kp = np.full(len(joint_order), 0.0)  # Default position gain for all joints
        self.kd = np.full(len(joint_order), 0.0)   # Default velocity gain for all joints

        # Get joint names
        self.joint_names = joint_order
        # Get body names
        self.body_names = None
        # Compute Ignore Joint Mask
        self.joint_limit_mask = torch.tensor([not (name in self.ignore_limit_joints) for name in joint_order], dtype=torch.bool)
        assert self.joint_limit_mask.ndim == 1
        assert self.joint_limit_mask.shape[0] == len(joint_order)

        # Set num joints
        self.num_joints = len(joint_order)

        # Initiate ROS2 node
        if init_rclpy:
            rclpy.init()
        Node.__init__(self, 'unitree_env')

        lowstate_topic = 'lowstate_buffer' if simulated_state else 'lowstate'
        # Create a subscriber to listen to an input topic (such as 'input_topic')
        self.lowstate_sub = self.create_subscription(
            LowState,  # Replace with your message type
            lowstate_topic,  # Replace with your input topic name
            self._lowstate_callback,
            1
        )

        # Create a publisher to publish to an output topic (such as 'output_topic')
        self.lowcmd_pub = self.create_publisher(
            LowCmd,  # Replace with your message type
            'lowcmd_buffer',  # Replace with your output topic name
            1
        )

        # Create command
        self.lowcmd = LowCmd()
        self.lowcmd_initialized = False

        # Create motor commands
        self.motor_cmd = []
        for id in range(self.num_joints):
            self.motor_cmd.append(MotorCmd(mode=1, reserve=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0))
        for id in range(self.num_joints, 35):
            self.motor_cmd.append(MotorCmd(mode=0, reserve=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0))
        self.lowcmd.motor_cmd = self.motor_cmd.copy()

        # Initialize state buffers
        self.joint_pos = torch.zeros(self.num_joints)
        self.joint_vel = torch.zeros(self.num_joints)
        self.root_rpy = torch.zeros(3)
        self.root_quat = torch.zeros(4)
        self.root_ang_vel = torch.zeros(3)

        # Initialize target positions
        self.target_positions = torch.zeros(self.num_joints)
        self.last_publish_time = time.monotonic()

        # Initialize CRC
        self.crc = CRC()

        # Initialize gamepad
        self.gamepad = Gamepad()
        self.gamepad_lstick = [0.0, 0.0]
        self.gamepad_rstick = [0.0, 0.0]
        self.gamepad_actions = ['gamepad.L1.pressed', 'gamepad.L2.pressed', 
                                'gamepad.R1.pressed', 'gamepad.R2.pressed', 
                                'gamepad.A.pressed', 'gamepad.B.pressed', 
                                'gamepad.X.pressed', 'gamepad.Y.pressed',
                                'gamepad.start.pressed', 'gamepad.select.pressed']

        # Initialize step frequency computation
        self.step_times = []
        self.max_record_steps = 50
        self.last_step_time = time.monotonic()

        # Get joint order
        self.joint_order_names = joint_order
        self.joint_order = list(range(len(joint_order)))

        self.action_joint_names = action_joint_names
        self.action_joints = []
        if action_joint_names is None or len(action_joint_names) == 0:
            self.action_joints = deepcopy(self.joint_order)
        else:
            for name in action_joint_names:
                self.action_joints.append(self.joint_order_names.index(name))
        
        print(f"UnitreeEnv initialized:")
        print(f"  Joints: {len(self.joint_order_names)}")
        print(f"  Control frequency: {control_freq} Hz")
        print(f"  Default PD gains: kp={self.kp[0]}, kd={self.kd[0]} (for all joints)")

    def _lowstate_callback(self, msg: LowState):
        """Callback for lowstate topic"""
        self.lowstate = msg
        if not self.lowcmd_initialized:
            self.lowcmd_initialized = True
            self.lowcmd.mode_pr = msg.mode_pr
            self.lowcmd.mode_machine = msg.mode_machine
        motor_cmds = [x for x in msg.motor_state]
        # assert len(motor_cmds) == self.num_joints, f"Expected {self.num_joints} motor commands, got {len(motor_cmds)}"

        self.gamepad.update(parse_remote_data(msg.wireless_remote))
        for action in self.gamepad_actions:
            if eval('self.' + action):
                if action in self.input_callbacks:
                    for callback in self.input_callbacks[action]:
                        callback()
        self.gamepad_lstick = [self.gamepad.lx, self.gamepad.ly]
        self.gamepad_rstick = [self.gamepad.rx, self.gamepad.ry]

        self.joint_pos[:] = torch.tensor([x.q for x in motor_cmds[:self.num_joints]]).float()
        self.joint_vel[:] = torch.tensor([x.dq for x in motor_cmds[:self.num_joints]]).float()
        self.root_rpy[:] = torch.tensor([msg.imu_state.rpy[0], msg.imu_state.rpy[1], msg.imu_state.rpy[2]]).float()    
        self.root_quat[:] = torch.tensor([msg.imu_state.quaternion[0], msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3]]).float()
        self.root_ang_vel[:] = torch.tensor([msg.imu_state.gyroscope[0], msg.imu_state.gyroscope[1], msg.imu_state.gyroscope[2]]).float()
        self._check_emergency_stop_condition()

    def _check_emergency_stop_condition(self) -> bool:
        """Check if the emergency stop condition is met"""
        emergency_stop = False
        if 'joint_pos_limit' in self.emergency_stop_condition:
            limit_factor = self.emergency_stop_condition['joint_pos_limit']
            exceed_high = (self.joint_pos >= self.joint_limits[:, 1] * limit_factor) & self.joint_limit_mask
            exceed_low = (self.joint_pos <= self.joint_limits[:, 0] * limit_factor) & self.joint_limit_mask
            any_exceed_high = torch.any(exceed_high).item()
            any_exceed_low = torch.any(exceed_low).item()
            if any_exceed_high or any_exceed_low:
                if any_exceed_high:
                    print(f"Joint position limit high exceeded: {exceed_high.nonzero().squeeze(-1)}")
                if any_exceed_low:
                    print(f"Joint position limit low exceeded: {exceed_low.nonzero().squeeze(-1)}")
                emergency_stop = True

        if 'roll_limit' in self.emergency_stop_condition:
            limit_factor = self.emergency_stop_condition['roll_limit']
            if self.root_rpy[0].abs().item() >= limit_factor:
                print(f"Root roll limit exceeded: {self.root_rpy[0].item()}")
                emergency_stop = True

        if emergency_stop:
            # emergency stop
            print(f"Emergency stop triggered")
            self._call_emergency_stop_hooks()
            if self.emergency_stop_breakpoint:
                self.target_positions[:] = self.joint_pos.clone()
                self.apply_pd_control()
                print(f"Emergency stop breakpoint triggered, exiting...")
                sys.exit(0)
        return emergency_stop

    def reset(self, fix_root=False):
        """
        Reset the robot to initial state
        
        Args:
            fix_root: If True, fix the root joint to make the robot static/floating
        """
        # no-ops in sim2real
        pass

    def refresh_data(self):
        """Refresh data"""
        # Refresh data by spinning once
        rclpy.spin_once(self, timeout_sec=self.spin_timeout)
    
    @BaseEnv.data_interface
    def get_joint_data(self):
        """
        Get current joint data
        
        Returns:
            dict: Dictionary containing joint positions, velocities, and accelerations
        """

        return {
            'joint_pos': self.joint_pos.clone(),  # Joint positions
            'joint_vel': self.joint_vel.clone(),  # Joint velocities
            'joint_cmd': self.target_positions.clone(),  # Joint commands
        }
    
    @BaseEnv.data_interface
    def get_root_data(self):
        """
        Get current root data
        """
        return {
            'root_rpy': self.root_rpy.clone(),  # Root euler (x, y, z)
            'root_quat': self.root_quat.clone(),  # Root orientation (quaternion)
            'root_ang_vel': self.root_ang_vel.clone(),  # Root angular velocity
        }
    
    @BaseEnv.data_interface
    def get_body_data(self):
        """
        Get current body data
        
        Returns:
            dict: Dictionary containing body positions, orientations, and velocities
        """
        # Get body positions and orientations
        raise NotImplementedError("Body data not implemented for sim2real, consider LiDAR plugin.")
    
    def set_pd_gains(self, kp=None, kd=None):
        """
        Set PD control gains
        
        Args:
            kp: Position gain(s). Can be:
                - scalar: applied to all joints
                - array: per-joint gains (length must match num_joints)
                - None: keeps current value
            kd: Velocity gain(s). Can be:
                - scalar: applied to all joints  
                - array: per-joint gains (length must match num_joints)
                - None: keeps current value
        """
        if kp is not None:
            if np.isscalar(kp):
                self.kp = np.full(self.num_joints, kp)
            else:
                if isinstance(kp, torch.Tensor):
                    kp = kp.cpu().numpy()
                elif isinstance(kp, list):
                    kp = np.array(kp)
                assert isinstance(kp, np.ndarray)
                if len(kp) == self.num_joints:
                    self.kp = kp.copy()
                elif len(kp) == len(self.joint_order):
                    self.kp[self.joint_order] = kp.copy()
                    assert isinstance(self.joint_order_names, list)
                    remain_joints = set(self.joint_names) - set(self.joint_order_names)
                    print(f"Remaining joints for kpkd: {remain_joints}")
                elif len(kp) == len(self.action_joints):
                    self.kp[self.action_joints] = kp.copy()
                    assert isinstance(self.action_joint_names, list)
                    remain_joints = set(self.joint_names) - set(self.action_joint_names)
                    print(f"Remaining joints for kpkd: {remain_joints}")
                else:
                    raise ValueError(f"Expected kp array of length {self.num_joints}, got {len(kp)}")
        
        if kd is not None:
            if np.isscalar(kd):
                self.kd = np.full(self.num_joints, kd)
            else:
                if isinstance(kd, torch.Tensor):
                    kd = kd.cpu().numpy()
                elif isinstance(kd, list):
                    kd = np.array(kd)
                assert isinstance(kd, np.ndarray)
                if len(kd) == self.num_joints:
                    self.kd = kd.copy()
                elif len(kd) == len(self.joint_order):
                    self.kd[self.joint_order] = kd.copy()
                elif len(kd) == len(self.action_joints):
                    self.kd[self.action_joints] = kd.copy()
                else:
                    raise ValueError(f"Expected kd array of length {self.num_joints}, got {len(kd)}")
                
        for i in range(self.num_joints):
            self.motor_cmd[i].kp = float(self.kp[i])
            self.motor_cmd[i].kd = float(self.kd[i])
        
        print(f"Set PD gains:")
        print(f"  kp: {self.kp}")
        print(f"  kd: {self.kd}")
    
    def get_pd_gains(self, return_full=False):
        """
        Get current PD gains
        
        Returns:
            tuple: (kp_array, kd_array) current PD gains for all joints
        """
        if return_full:
            return torch.from_numpy(self.kp.copy()).float(), torch.from_numpy(self.kd.copy()).float()
        else:
            return torch.from_numpy(self.kp[self.joint_order].copy()).float(), torch.from_numpy(self.kd[self.joint_order].copy()).float()
    
    def apply_pd_control(self):
        """Apply PD control using current target positions"""
        # For position actuators, we just set the target positions
        # MuJoCo will automatically apply PD control
        for i in range(self.num_joints):
            self.motor_cmd[i].q = self.target_positions[i].item()
        self.lowcmd.motor_cmd = self.motor_cmd.copy()
        self.lowcmd.crc = self.crc.Crc(self.lowcmd) # type: ignore
        self.lowcmd_pub.publish(self.lowcmd)

    @property
    def step_frequency(self):
        """Compute step frequency"""
        if len(self.step_times) == 0:
            return self.control_freq
        return 1.0 / np.mean(self.step_times)

    def step_complete(self):
        """Check if the simulation step is complete"""
        step_complete = time.monotonic() - self.last_publish_time > self.control_dt - self.release_time_delta
        if step_complete:
            rclpy.spin_once(self, timeout_sec=self.spin_timeout)
        return step_complete
    
    def step(self, actions=None):
        """
        Step the simulation forward by running decimation number of simulation steps
        
        Args:
            actions: Optional array of target positions for joints (excluding root)
                    If provided, updates the target positions before applying control
        Returns:
            bool: True if simulation is still running, False if it should stop
        """
        # Update target positions if provided
        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            elif isinstance(actions, list):
                actions = torch.tensor(actions)
            assert isinstance(actions, torch.Tensor)
            if len(actions) != len(self.action_joints):
                raise ValueError(f"Expected actions array of length {len(self.action_joints)}, got {len(actions)}")
            self.target_positions[self.action_joints] = actions.clone().float().cpu()

        # Update step count
        self.step_count += 1
        
        # Apply PD control
        self.apply_pd_control()
        self.last_publish_time = time.monotonic()

        # Compute step frequency
        self.step_times.append(time.monotonic() - self.last_step_time)
        self.last_step_time = time.monotonic()
        if len(self.step_times) > self.max_record_steps:
            self.step_times.pop(0)

        if self.align_time:
            frequency = self.step_frequency
            if frequency > self.control_freq + self.align_tolerance:
                self.release_time_delta -= self.align_step_size
            elif frequency < self.control_freq - self.align_tolerance:
                self.release_time_delta += self.align_step_size
            self.release_time_delta = max(0.0, self.release_time_delta)
            self.release_time_delta = min(self.control_dt, self.release_time_delta)
        return True
    
    def run_simulation(self, max_steps=None):
        """
        Run the simulation loop
        
        Args:
            max_steps: Maximum number of steps to run (None for infinite)
        """
        # Error in sim2real
        raise NotImplementedError("Run simulation not implemented for sim2real")
    
    def close(self):
        """Close the environment and cleanup resources"""
        # Error in sim2real
        self.destroy_node()

def main():
    """Main function to run the sim2real"""
    rclpy.init()

    joint_names = [
        "left_hip_yaw_joint",
        "left_hip_pitch_joint", 
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint", 
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint", 
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_wrist_pitch_joint", 
        "right_wrist_yaw_joint",
    ]

    # Create environment
    env = UnitreeEnv(
        control_freq=50,
        joint_order=joint_names,
    )
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

