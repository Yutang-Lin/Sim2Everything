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

def reindex(from_order: list[str], to_order: list[str]):
    """
    Reindex the order of the list from the from_order to the to_order.
    """
    return [from_order.index(item) for item in to_order]

import torch
class AutoQueue:
    """
    A fixed-size queue implementation for PyTorch tensors with automatic rolling.
    
    This class provides a circular buffer for storing PyTorch tensors of a fixed shape.
    When the queue is full and a new item is pushed, the oldest item is automatically
    removed and the new item is added to the front. All operations are performed in
    inference mode for efficiency.
    
    The queue maintains tensors on the same device as the example item provided during
    initialization, ensuring efficient GPU/CPU operations without device transfers.
    
    Attributes:
        max_size (int): Maximum number of items the queue can hold
        queue (torch.Tensor): The underlying tensor storage with shape (max_size, *item_shape)
    
    Args:
        example_item (torch.Tensor): Example tensor that defines the shape and device
            for all items in the queue. The queue will store tensors of this exact shape.
        max_size (int): Maximum number of items the queue can hold. Must be positive.
    
    Example:
        >>> # Create a queue for 2D position vectors
        >>> pos_tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> queue = AutoQueue(pos_tensor, max_size=5)
        >>> 
        >>> # Push new positions
        >>> queue.push(torch.tensor([4.0, 5.0, 6.0]))
        >>> queue.push(torch.tensor([7.0, 8.0, 9.0]))
        >>> 
        >>> # Get all stored positions
        >>> all_positions = queue.get_tensor()  # Shape: (5, 3)
        >>> print(all_positions[0])  # Most recent: [7.0, 8.0, 9.0]
        >>> print(all_positions[1])  # Previous: [4.0, 5.0, 6.0]
    """
    def __init__(self, example_item: torch.Tensor, max_size: int):
        self.max_size = max_size
        self.queue = torch.zeros(max_size, *example_item.shape, device=example_item.device)
    
    @torch.inference_mode()
    def push(self, item: torch.Tensor):
        self.queue = self.queue.roll(shifts=1, dims=0)
        self.queue[0] = item.clone()
    
    @torch.inference_mode()
    def get_tensor(self) -> torch.Tensor:
        return self.queue.clone()

    @torch.inference_mode()
    def clear(self):
        self.queue.fill_(0.0)

import joblib
import pytorch_kinematics as pk
import torch
import numpy as np
import tqdm
import math_utils
import time
import pytorch_kinematics.transforms as transforms
from copy import deepcopy
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F

@torch.inference_mode()
def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """
    Compute angular velocity from adjacent quaternions (w, x, y, z):
    - Relative rotation q_rel = inv(q_prev) * q_next
    - Extract rotation angle and axis from q_rel
    - Return (angle / dt) * axis
    """
    q_inv = transforms.quaternion_invert(q_prev)
    q_rel = transforms.quaternion_multiply(q_inv, q_next)

    w = torch.clamp(q_rel[..., 0:1], -1.0, 1.0)
    angle = 2.0 * torch.arccos(w)
    sin_half = torch.sqrt(1.0 - w*w)

    axis = q_rel[..., 1:] / (sin_half + eps)
    return (angle / dt) * axis * (sin_half > eps)

from typing import Literal
class MotionSwitcher:
    """
    A motion switching system for robotic character animation with smooth transitions.
    
    This class manages multiple motion sequences and provides smooth interpolation between
    them. It supports forward kinematics computation, motion normalization, and real-time
    motion switching with configurable transition periods.
    
    The system operates as a state machine with three states:
    - 'idle': Ready to play or switch motions
    - 'playing': Currently playing a motion sequence
    - 'switching': Transitioning between two motions with interpolation
    
    Attributes:
        device (torch.device): PyTorch device (CUDA if available, else CPU)
        motion_file (dict): Loaded motion data from pickle file
        motions (list): List of available motion names
        current_motion (int): Index of currently active motion
        chain (pk.Chain): Forward kinematics chain built from URDF
        joint_names (list[str]): Names of robot joints
        eef_links (list[str]): Names of end-effector links
        motion_quat_convention (str): Quaternion convention ("xyzw" or "wxyz")
        num_futures (int): Number of future frames to compute
        future_dt (float): Time step between future frames
        switch_interval (float): Duration for motion transitions (seconds)
        root_quat (torch.Tensor): Current root orientation quaternion
    
    Args:
        motion_path (str): Path to pickle file containing motion data
        urdf_path (str): Path to URDF file for robot model
        joint_names (list[str]): List of joint names in the robot
        eef_links (list[str]): List of end-effector link names
        motion_quat_convention (Literal["xyzw", "wxyz"], optional): Quaternion 
            convention used in motion data. Defaults to "wxyz".
        switch_interval (float, optional): Duration for smooth motion transitions 
            in seconds. Defaults to 5.0.
        num_futures (int, optional): Number of future motion frames to compute 
            for prediction. Defaults to 5.
        future_dt (float, optional): Time step between future frames in seconds. 
            Defaults to 0.1.
    
    Example:
        >>> switcher = MotionSwitcher(
        ...     motion_path="motions.pkl",
        ...     urdf_path="robot.urdf", 
        ...     joint_names=["joint1", "joint2"],
        ...     eef_links=["left_hand", "right_hand"]
        ... )
        >>> switcher.play_current_motion()
        >>> switcher.switch_motion(1)  # Switch to motion 1
        >>> joint_pos, rb_pos, rb_quat, rb_lin_vel, rb_ang_vel = switcher.get_motion_data()
    """
    def __init__(self, motion_path: str, 
                 urdf_path: str,
                 joint_names: list[str], 
                 eef_links: list[str],
                 motion_quat_convention: Literal["xyzw", "wxyz"] = "wxyz",
                 switch_interval: float = 5.0,
                 num_futures: int = 5,
                 future_dt: float = 0.1,
                 initial_motion_id: int = 0,
                 initial_motion_time: float = 0.0,
                 motion_progress_bar: bool = False,
                 upsample_fps: int = 50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.motion_file = joblib.load(motion_path)
        self.motions = list(self.motion_file.keys())
        for i, name in enumerate(self.motions):
            print(f"{i}: {name}")
        self.current_motion = initial_motion_id
        self._next_motion = -1
        self._delta_motion_time = initial_motion_time
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=self.device)
        self.joint_names = joint_names
        self.eef_links = eef_links
        self.motion_quat_convention = motion_quat_convention
        self.upsample_fps = upsample_fps
        # (kx, ky) for kernel size
        self.gaussian_blur = GaussianBlur(kernel_size=(5, 1), sigma=(1.0, 1.0))
        self._build_motion_data()

        self.num_futures = num_futures
        self.future_dt = future_dt
        self.switch_interval = switch_interval
        self._state = 'idle' # state machine
        self._ready_to_play = False
        self._want_to_play = False
        self._dynamic = False
        self.motion_progress_bar = motion_progress_bar
        self._motion_progress_bar = None

        # compute initial motion data
        self.current_motion_data = self._compute_motion_data(self.current_motion, initial_motion_time)

    @torch.inference_mode()
    def gaussian_filter(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], -1).permute(1, 0)[:, None, None, :]
        x = self.gaussian_blur(x)[:, 0, 0].permute(1, 0).reshape(*x_shape)
        return x

    @torch.inference_mode()
    def upsample_motion_data(self, motion_data, motion_fps, new_fps):
        motion_length = motion_data['joint_pos'].shape[0]
        new_length = int(motion_length * new_fps / motion_fps)
        new_motion_data = {}
        new_motion_data['joint_pos'] = F.interpolate(motion_data['joint_pos'][None, :].transpose(1, 2), size=(new_length,), mode='linear')[0].transpose(0, 1)
        new_motion_data['root_pos'] = F.interpolate(motion_data['root_pos'][None, :].transpose(1, 2), size=(new_length,), mode='linear')[0].transpose(0, 1)
        new_motion_data['root_quat'] = F.interpolate(motion_data['root_quat'][None, :].transpose(1, 2), size=(new_length,), mode='linear')[0].transpose(0, 1)
        new_motion_data['root_quat'] = new_motion_data['root_quat'] / (new_motion_data['root_quat'].norm(dim=-1, keepdim=True) + 1e-6)
        new_motion_data['fps'] = new_fps
        new_motion_data['length'] = new_length
        new_motion_data['length_s'] = new_length / new_fps
        return new_motion_data

    def _build_motion_data(self):
        for id, motion in enumerate(tqdm.tqdm(self.motions)):
            raw_motion_data = self.motion_file[motion]
            motion_data = dict()
            motion_data['joint_pos'] = torch.from_numpy(raw_motion_data['dof'] if 'dof' in raw_motion_data else raw_motion_data['joint_pos']).float().to(self.device)
            motion_data['root_pos'] = torch.from_numpy(raw_motion_data['root_trans_offset'] if 'root_trans_offset' in raw_motion_data else raw_motion_data['root_pos']).float().to(self.device)
            motion_data['root_quat'] = torch.from_numpy(raw_motion_data['root_rot'] if 'root_rot' in raw_motion_data else raw_motion_data['root_quat']).float().to(self.device)
            if self.motion_quat_convention == "xyzw":
                motion_data['root_quat'] = math_utils.convert_quat(motion_data['root_quat'], to="wxyz")
            motion_data = self.upsample_motion_data(motion_data, raw_motion_data['fps'], self.upsample_fps)

            motion_joint_vel = torch.gradient(motion_data['joint_pos'], spacing=1/motion_data['fps'], dim=0)[0]
            motion_joint_vel = self.gaussian_filter(motion_joint_vel)
            all_joint_pos = {k: motion_data['joint_pos'][:, i] for i, k in enumerate(self.joint_names)}
            fk_result = self.chain.forward_kinematics(all_joint_pos)

            root_pos = motion_data['root_pos']
            initial_root_pos = root_pos[0:1].clone()
            initial_root_pos[:, 2] = 0.0
            root_pos = root_pos - initial_root_pos
            root_quat = motion_data['root_quat']
            root_quat_yaw_inv = math_utils.quat_conjugate(math_utils.yaw_quat(root_quat[0:1]))
            root_pos = transforms.quaternion_apply(root_quat_yaw_inv, root_pos)
            root_quat = transforms.quaternion_multiply(root_quat_yaw_inv, root_quat)

            eef_matrices = [fk_result[link].get_matrix() for link in self.eef_links]
            rb_pos = transforms.quaternion_apply(root_quat.unsqueeze(1), torch.stack([eef_matrix[:, :3, 3] for eef_matrix in eef_matrices], dim=1)) + root_pos.unsqueeze(1)
            rb_quat = transforms.quaternion_multiply(root_quat.unsqueeze(1), torch.stack([math_utils.quat_from_matrix(eef_matrix[:, :3, :3]) for eef_matrix in eef_matrices], dim=1))
            rb_lin_vel = torch.gradient(rb_pos, spacing=1/motion_data['fps'], dim=0)[0]
            rb_lin_vel = self.gaussian_filter(rb_lin_vel)
            rb_ang_vel = compute_angular_velocity(rb_quat[:-1], rb_quat[1:], 1/motion_data['fps'])
            rb_ang_vel = torch.cat([rb_ang_vel, rb_ang_vel[-1:]], dim=0)
            rb_ang_vel = self.gaussian_filter(rb_ang_vel)
            rb_ang_vel = transforms.quaternion_apply(rb_quat, rb_ang_vel)

            motion_data['joint_pos'] = motion_data['joint_pos']
            motion_data['root_pos'] = motion_data['root_pos']
            motion_data['root_quat'] = motion_data['root_quat']
            motion_data['joint_vel'] = motion_joint_vel
            motion_data['rb_pos'] = rb_pos
            motion_data['rb_quat'] = rb_quat
            motion_data['rb_lin_vel'] = rb_lin_vel
            motion_data['rb_ang_vel'] = rb_ang_vel
            self.motion_file[motion] = motion_data
            self.normalize_motion(id, motion_data['root_quat'][0])
            for k, v in motion_data.items():
                if isinstance(v, torch.Tensor):
                    motion_data[k] = v.cpu()

    def _compute_motion_data_at_time(self, motion_id, time):
        motion_data = self.motion_file[self.motions[motion_id]]
        motion_frac = time * motion_data['fps']
        frame_floor = min(int(motion_frac), motion_data['length'] - 1)
        frame_ceil = min(int(motion_frac + 1), motion_data['length'] - 1)
        ratio = motion_frac - frame_floor
        motion_joint_pos = motion_data['joint_pos'][frame_floor] * (1 - ratio) + motion_data['joint_pos'][frame_ceil] * ratio
        motion_rb_pos = motion_data['rb_pos'][frame_floor] * (1 - ratio) + motion_data['rb_pos'][frame_ceil] * ratio
        motion_rb_quat = motion_data['rb_quat'][frame_floor] * (1 - ratio) + motion_data['rb_quat'][frame_ceil] * ratio
        motion_rb_lin_vel = motion_data['rb_lin_vel'][frame_floor] * (1 - ratio) + motion_data['rb_lin_vel'][frame_ceil] * ratio
        motion_rb_ang_vel = motion_data['rb_ang_vel'][frame_floor] * (1 - ratio) + motion_data['rb_ang_vel'][frame_ceil] * ratio
        return motion_joint_pos, motion_rb_pos, motion_rb_quat, motion_rb_lin_vel, motion_rb_ang_vel

    def _compute_motion_data(self, motion_id, time):
        futures = [time + i * self.future_dt for i in range(self.num_futures)]
        datas = [self._compute_motion_data_at_time(motion_id, future) for future in futures]
        results = [torch.stack(data, dim=0) for data in zip(*datas)]
        return results

    def get_motion_data(self):
        return self.current_motion_data[0], self.current_motion_data[1], self.current_motion_data[2], self.current_motion_data[3] * float(self._dynamic), self.current_motion_data[4] * float(self._dynamic)

    def normalize_motion(self, motion_id, initial_quat: torch.Tensor):
        if motion_id >= len(self.motions) or motion_id < 0:
            print(f"Invalid motion id {motion_id}, switching to first motion")
            motion_id = 0
        motion_data = self.motion_file[self.motions[motion_id]]
        delta_quat = math_utils.quat_mul(math_utils.yaw_quat(initial_quat), math_utils.quat_conjugate(math_utils.yaw_quat(motion_data['rb_quat'][0, 0])))
        motion_data['rb_pos'] = transforms.quaternion_apply(delta_quat[None, None, :], motion_data['rb_pos'])
        motion_data['rb_quat'] = transforms.quaternion_multiply(delta_quat[None, None, :], motion_data['rb_quat'])
        motion_data['rb_lin_vel'] = transforms.quaternion_apply(delta_quat[None, None, :], motion_data['rb_lin_vel'])
        motion_data['rb_ang_vel'] = transforms.quaternion_apply(delta_quat[None, None, :], motion_data['rb_ang_vel'])
        self.motion_file[self.motions[motion_id]] = motion_data

    def play_current_motion(self):
        if not self._ready_to_play:
            self._want_to_play = False
            print("Not ready to play")
            return
        self._want_to_play = True
        self._start_play_time = time.monotonic()

    def switch_motion(self, to_motion_id: int):
        if self._state != 'idle':
            print("Not in idle state, cannot switch motion")
            return
        if to_motion_id >= len(self.motions) or to_motion_id < 0:
            print(f"Invalid motion id {to_motion_id}, switching to first motion")
            to_motion_id = 0
        self._next_motion = to_motion_id

    def _interpolate_motion_data(self, last_data, next_data, ratio):
        interpolated_data = []
        for last, next in zip(last_data, next_data):
            interpolated_data.append(last * (1 - ratio) + next * ratio)
        interpolated_data[2] = interpolated_data[2] / (interpolated_data[2].norm(dim=-1, keepdim=True) + 1e-6)
        return tuple(interpolated_data)

    def _idle_update(self):
        self._ready_to_play = True
        if self._next_motion != -1:
            self._state = 'switching'
            print(f"Starting to switch to motion {self._next_motion}")
            self._start_switch_time = time.monotonic()
            self._last_motion_data = deepcopy(self.current_motion_data)
            self._next_motion_data = self._compute_motion_data(self._next_motion, self._delta_motion_time)
        if self._want_to_play:
            self._state = 'playing'
            print(f"Starting to play motion {self.current_motion}")
            self._start_play_time = time.monotonic()
            self._want_to_play = False
            if self.motion_progress_bar:
                self._motion_progress_bar = tqdm.tqdm(total=self.motion_file[self.motions[self.current_motion]]['length_s'], 
                                                        desc=f"Playing motion {self.motions[self.current_motion]}",
                                                        initial=round(self._delta_motion_time, 3))

    def _switching_update(self):
        self._ready_to_play = False
        switch_time = time.monotonic() - self._start_switch_time
        if switch_time > self.switch_interval:
            self._state = 'idle'
            print(f"Finished switching to motion {self._next_motion}")
            self.current_motion = self._next_motion
            self._next_motion = -1
            self._delta_motion_time = 0.0
            self.current_motion_data = self._next_motion_data
        else:
            ratio = switch_time / self.switch_interval
            interpolated_data = self._interpolate_motion_data(self._last_motion_data, self._next_motion_data, ratio)
            self.current_motion_data = interpolated_data

    def _playing_update(self):
        self._ready_to_play = False
        self._dynamic = True
        play_time = time.monotonic() - self._start_play_time + self._delta_motion_time
        if self.motion_progress_bar:
            self._motion_progress_bar.update(round(play_time - self._motion_progress_bar.n, 3))
        if play_time > self.motion_file[self.motions[self.current_motion]]['length_s']:
            self._state = 'idle'
            self._dynamic = False
            print("Finished playing motion")
            if self.motion_progress_bar:
                self._motion_progress_bar.close()
            return
        self.current_motion_data = self._compute_motion_data(self.current_motion, play_time)

    def update(self):
        if self._state == 'idle':
            self._idle_update()
        elif self._state == 'playing':
            self._playing_update()
        elif self._state == 'switching':
            self._switching_update()

    def run(self, frequency: int = 50):
        step_dt = 1.0 / frequency
        last_time = time.monotonic()
        while True:
            while time.monotonic() - last_time < step_dt:
                time.sleep(0.001)
            self.update()
            