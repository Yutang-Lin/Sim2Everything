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
from .sim2real import UnitreeEnv
from vla_data_collection.head_servo_ctrl import G1HeadCtrlNode

class UnitreeHeadEnv(UnitreeEnv, Node):
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
        UnitreeEnv.__init__(self, 
                        control_freq=control_freq,
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
        self.joint_temperature = torch.zeros(self.num_joints)
        self.tick = 0
        self.head_ctrl_node = G1HeadCtrlNode()
        self.head_ctrl_node.ctrl_head(0., 0.)
        self.max_action_factor = torch.ones(29)
        
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

        self.joint_temperature[:] = torch.tensor([x.temperature[1] for x in motor_cmds[:self.num_joints]]).float()
        self.joint_pos[:] = torch.tensor([x.q for x in motor_cmds[:self.num_joints]]).float()
        self.joint_vel[:] = torch.tensor([x.dq for x in motor_cmds[:self.num_joints]]).float()
        self.root_rpy[:] = torch.tensor([msg.imu_state.rpy[0], msg.imu_state.rpy[1], msg.imu_state.rpy[2]]).float()    
        self.root_quat[:] = torch.tensor([msg.imu_state.quaternion[0], msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3]]).float()
        self.root_ang_vel[:] = torch.tensor([msg.imu_state.gyroscope[0], msg.imu_state.gyroscope[1], msg.imu_state.gyroscope[2]]).float()
        self.tick = msg.tick
        self._check_emergency_stop_condition()

    def get_state_tick(self):
        """Get current state tick"""
        return self.tick
    
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
        # For PD actuators, we just set the target positions
        # if clip_action_to_torque_limit is True, we clip the target positions to the torque limits
        if self.clip_action_to_torque_limit:
            p_term = (self.target_positions - self.joint_pos) * self.kp
            d_term = (0.0 - self.joint_vel) * self.kd
            tau_est = p_term + d_term
            tau_est = torch.clamp(tau_est, -self.torque_limits, self.torque_limits)
            self.target_positions[:] = (tau_est - d_term) / self.kp + self.joint_pos # clip to torque limits

        # joint overheat protection
        self.max_action_factor[((self.joint_temperature > 80) & (self.joint_temperature <= 100)).tolist()] = 0.5
        self.max_action_factor[(self.joint_temperature > 100).tolist()] = 0.0
        self.max_action_factor[(self.joint_temperature <= 80).tolist()] = 1.0
        self.target_positions = self.joint_pos + (self.target_positions - self.joint_pos) * self.max_action_factor

        # publish motor commands
        for i in range(self.num_joints):
            self.motor_cmd[i].q = self.target_positions[i].item()
        self.lowcmd.motor_cmd = self.motor_cmd.copy()
        self.lowcmd.crc = self.crc.Crc(self.lowcmd) # type: ignore
        self.lowcmd_pub.publish(self.lowcmd)
    
    def get_head_state(self):
        pitch_degree, yaw_degree = self.head_ctrl_node.get_head_state()
        return pitch_degree, yaw_degree

    def get_joint_temperature(self):
        return self.joint_temperature.clone()

    def step(self, actions=None, tgt_pitch=None, tgt_yaw=None):
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

        # import pdb; pdb.set_trace()
        if tgt_pitch is not None and tgt_yaw is not None:
            if isinstance(tgt_pitch, torch.Tensor):
                tgt_pitch = float(tgt_pitch.detach().cpu())
                tgt_yaw = float(tgt_yaw.detach().cpu())
            elif isinstance(tgt_pitch, list):
                tgt_pitch = float(tgt_pitch[0])
                tgt_yaw = float(tgt_yaw[0])
            elif isinstance(tgt_pitch, np.ndarray):
                tgt_pitch = float(tgt_pitch)
                tgt_yaw = float(tgt_yaw)
            else:
                tgt_pitch = float(tgt_pitch)
                tgt_yaw = float(tgt_yaw)

            self.head_ctrl_node.ctrl_head(tgt_pitch, tgt_yaw)
        # Update step count
        self.step_count += 1

        # Apply PD control
        self.apply_pd_control()
        # Compute step frequency
        self.step_times.append(time.monotonic() - self.last_publish_time)
        # print(f"Step time: {self.step_times[-1]}")
        # Update last publish time
        self.last_publish_time = time.monotonic()

        if len(self.step_times) > self.max_record_steps:
            self.step_times.pop(0)

        if self.align_time:
            frequency = self.step_frequency
            # print(f"Frequency: {frequency}")
            if frequency > self.control_freq + self.align_tolerance:
                self.release_time_delta -= self.align_step_size
            elif frequency < self.control_freq - self.align_tolerance:
                self.release_time_delta += self.align_step_size
            # print(f"Release time delta: {self.release_time_delta}")
            self.release_time_delta = max(0.0, self.release_time_delta)
            self.release_time_delta = min(self.control_dt, self.release_time_delta)
        return True

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
    env = UnitreeHeadEnv(
        control_freq=50,
        joint_order=joint_names,
    )
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

