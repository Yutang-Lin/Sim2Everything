# Sim-to-Everything
## Introduction
Simple implementation of Gym style simulation environments for sim2sim. Along with a wrapped ROS2 environment for Unitree robots' real deployment with same interface as simulators, provide seamless transfer between sim2sim and sim2real.

## Roadmap
Here shows the roadmap of this repository.
- [x] Implement Mujoco as basic sim2sim environment.
- [x] Implement ROS2 environment for unitree sim2real.
- [ ] Implement SAPIEN sim2sim environment for tasks need visual informations.

## Usage
Format your server as following to get seamlessly transfer from sim2sim to sim2real, and wrapped ROS2 communication.
```python
"""import packages"""
from deploy_utils.sim2sim import MujocoEnv
# from deploy_utils.sim2real import UnitreeEnv # for sim2real
from deploy_utils.utils import reindex # reindex if needed

"""define joint order if needed"""
policy_joint_order = [...]
real_joint_order = [...]
policy_to_real = reindex(policy_joint_order, real_joint_order) # reindex(from, to)

"""initialize env, setup control hz etc."""
env = MujocoEnv(
    control_freq=50,
    xml_path='...', # sim2sim only
    simulation_freq=1000 # sim2sim only
    joint_order=real_joint_order
) # UnitreeEnv(...) # switch freely

"""control loop for your policy"""
while not end_condition:
    while not env.step_complete():
        time.sleep(0.001)
    env.refresh_data()

    joint_data = env.get_joint_data() # dict of joint_pos, joint_vel, joint_cmd
    root_data = env.get_root_data() # dict of root_rpy, root_quat, root_ang_vel
    # body_data = env.get_body_data() # dict of body_pos ..., only in sim2sim

    obs = compute_observation(joint_data, root_data)
    actions = policy(obs.view(1, -1)).view(-1)[policy_to_real] # compute and reindex
    actions = (actions - default_actions) * action_scale # etc.

    env.step(actions)

    """for sim2sim resets"""
    if terminate_condition:
        env.reset(fix_root=True) # reset as you needed in sim2sim scenarios
```
