import torch
import numpy as np
from typing import Callable, Any
import threading
import sys
import select
import time

def _noise_like(x: torch.Tensor, noise_type: str = 'uniform') -> torch.Tensor:
    if noise_type == 'uniform':
        return torch.rand_like(x) * 2 - 1
    elif noise_type == 'gaussian':
        return torch.randn_like(x)
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

class BaseEnv:
    simulated = True

    def __init__(self, control_freq: int = 100, 
                 joint_order: list[str] | None = None,
                 action_joint_names: list[str] | None = None,
                 joint_limits: list[tuple[float, float]] | np.ndarray | None = None,
                 release_time_delta: float = 0.0,
                 align_time: bool = True,
                 align_step_size: float = 0.00005,
                 align_tolerance: float = 2.0,
                 init_rclpy: bool = True,
                 spin_timeout: float = 0.001,
                 launch_input_thread: bool = True,
                 simulated_state: bool = False,
                 emergency_stop_condition: dict[str, Any] = {
                    'joint_pos_limit': 0.98,
                    'ignore_limit_joints': [],
                    'roll_limit': 1.57,
                 },
                 emergency_stop_breakpoint: bool = True,

                 # Simulation only
                 model_path: str = '', 
                 simulation_freq: int = 1000,
                 joint_armature: float = 0.01, 
                 joint_damping: float = 0.1, 
                 enable_viewer: bool = True,
                 enable_ros_control: bool = False,
                 noise_level: float = 0.0,
                 noise_type: str = 'uniform',
                 noise_scales: dict[str, float] = {
                        'joint_pos': 0.01,
                        'joint_vel': 1.50,
                        'root_rpy': 0.1,
                        'root_quat': 0.05,
                        'root_ang_vel': 0.2,
                    }
                 ):
        self.noise_level: float = noise_level
        self.noise_type: str = noise_type
        self.noise_scales: dict[str, float] = noise_scales
        self.input_callbacks: dict[str, Callable] = {}
        self.terminal_input_callbacks: list[Callable] = []
        self.launch_input_thread: bool = launch_input_thread
        self.joint_limits: list[tuple[float, float]] | np.ndarray | None = joint_limits

        # Emergency stop hooks
        self.emergency_stop_hooks: list[Callable] = []
        self.emergency_stop_condition: dict[str, Any] = emergency_stop_condition
        self.emergency_stop_breakpoint: bool = emergency_stop_breakpoint

        # Launch input thread if enabled
        if self.launch_input_thread:
            print('[INFO]: Launching input thread, terminal input available')
            self.input_thread = threading.Thread(target=self._input_thread)
            self.input_thread.start()
        else:
            self.input_thread = None

    @staticmethod
    def data_interface(func: Callable) -> Callable:
        """marking the function as a data interface"""
        def wrapper(self, *args, **kwargs):
            data = func(self, *args, **kwargs)
            data = self._data_interface(data)
            return data
        def str(*args, **kwargs):
            return f"WrappedInterface({func.__str__(*args, **kwargs)})"
        def repr(*args, **kwargs):
            return f"WrappedInterface({func.__repr__(*args, **kwargs)})"
        def format(*args, **kwargs):    
            return f"WrappedInterface({func.__format__(*args, **kwargs)})"
        wrapper.__str__ = str
        wrapper.__repr__ = repr
        wrapper.__format__ = format
        return wrapper
    
    def register_input_callback(self, key: str, callback: Callable) -> None:
        """Register a callback for a specific key"""
        if key not in self.input_callbacks:
            self.input_callbacks[key] = []
        self.input_callbacks[key].append(callback)

    def register_terminal_input_callback(self, callback: Callable) -> None:
        """Register a callback for terminal input"""
        self.terminal_input_callbacks.append(callback)

    def register_emergency_stop_hook(self, hook: Callable) -> None:
        """Register an emergency stop hook"""
        self.emergency_stop_hooks.append(hook)

    def _call_emergency_stop_hooks(self) -> None:
        """Call all emergency stop hooks"""
        for hook in self.emergency_stop_hooks:
            hook()

    def _input_thread(self):
        """Input thread"""
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input in self.input_callbacks:
                    for callback in self.input_callbacks[user_input]:
                        callback()
                for callback in self.terminal_input_callbacks:
                    callback(user_input)
            time.sleep(0.01)

    def reset(self, fix_root: bool = False) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")

    def step_complete(self) -> bool:
        raise NotImplementedError("This function should be implemented by the subclass")

    def step(self, actions=None) -> bool:
        raise NotImplementedError("This function should be implemented by the subclass")

    def refresh_data(self) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")

    @data_interface
    def get_joint_data(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This function should be implemented by the subclass")

    @data_interface
    def get_root_data(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This function should be implemented by the subclass")
    
    def set_pd_gains(self, kp: torch.Tensor | np.ndarray | list[float], 
                     kd: torch.Tensor | np.ndarray | list[float]) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")

    def get_pd_gains(self, return_full: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This function should be implemented by the subclass")

    def apply_pd_control(self) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")

    def close(self) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")
    
    def _data_interface(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._apply_noise(data)

    # Simulation only functions
    def _apply_noise(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.simulated or self.noise_level == 0.0:
            return data
        for key, value in data.items():
            if key in self.noise_scales:
                data[key] = value + _noise_like(value, self.noise_type) * self.noise_scales[key] * self.noise_level
        return data

    def run_simulation(self, max_steps: int | None = None) -> None:
        raise NotImplementedError("This function should be implemented by the subclass")

    @data_interface
    def get_body_data(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("This function should be implemented by the subclass")
