import torch
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as transforms
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
import mujoco
import mujoco.viewer

class FKService:
    def __init__(self, 
                urdf_path: str, 
                joint_names: list[str],
                body_names: list[str],
                xml_path: str | None = None,
                enable_viewer: bool = False,
                max_viewer_spheres: int = 0):
        with open(urdf_path, 'r') as f:
            urdf_string = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_string)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.body_names = body_names

        self.joint_ids = [self.joint_names.index(name) for name in joint_names]
        self.joint_pos = torch.zeros(len(self.joint_names))
        self.root_pos = torch.zeros(3)
        self.root_quat = torch.zeros(4)

        self.fk_rb_pos = torch.zeros(len(self.body_names), 3)
        self.fk_rb_quat = torch.zeros(len(self.body_names), 4)

        if xml_path is not None and enable_viewer:
            self.enable_mujoco = True
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

            self.mj_joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)][1:]
            self.mj_joint_pos = np.zeros(7+len(self.mj_joint_names))
            self.mj_joint_ids = [7 + self.mj_joint_names.index(name) for name in joint_names]

            self.mj_body_names = [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]
            self.mj_body_ids = [self.mj_body_names.index(name) for name in body_names]
        else:
            self.enable_mujoco = False

        if max_viewer_spheres > 0:
            assert enable_viewer, "enable_viewer must be True when max_viewer_spheres > 0"
            self.debug_spheres = [self.add_visual_sphere(pos=np.zeros(3), radius=0.03, rgba=(1, 1, 0, 1)) for _ in range(max_viewer_spheres)]

    def add_visual_sphere(self, pos, radius=0.01, rgba=(1, 0, 0, 1)):
        """Add a visual sphere to the viewer"""
        if self.mj_viewer.user_scn.ngeom >= self.mj_viewer.user_scn.maxgeom:
            return
        self.mj_viewer.user_scn.ngeom += 1  # increment ngeom
        if isinstance(rgba, torch.Tensor):
            rgba = rgba.cpu().numpy()
        elif isinstance(rgba, (list, tuple)):
            rgba = np.array(rgba)
        assert isinstance(rgba, np.ndarray)
        assert rgba.shape == (4,)
        # initialise a new capsule, add it to the scene using mjv_makeConnector
        mujoco.mjv_initGeom(self.mj_viewer.user_scn.geoms[self.mj_viewer.user_scn.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
        mujoco.mjv_connector(self.mj_viewer.user_scn.geoms[self.mj_viewer.user_scn.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            pos, pos + 1e-3)
        return self.mj_viewer.user_scn.geoms[self.mj_viewer.user_scn.ngeom-1]

    @torch.inference_mode()
    def update_root_state(self, root_pos: torch.Tensor, root_quat: torch.Tensor):
        self.root_pos[:] = root_pos
        self.root_quat[:] = root_quat

    @torch.inference_mode()
    def update_debug_spheres(self, pos: np.ndarray, rgba: np.ndarray):
        if self.debug_spheres is not None:
            for i in range(len(self.debug_spheres)):
                self.debug_spheres[i].pos = pos[i, :3]
                self.debug_spheres[i].rgba = rgba[i, :4]

    @torch.inference_mode()
    def forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if not self.enable_mujoco:
            self.joint_pos[self.joint_ids] = joint_pos
            fk_result = self.chain.forward_kinematics(self.joint_pos)
            for i, k in enumerate(self.body_names):
                matrix = fk_result[k].get_matrix()[0]
                self.fk_rb_pos[i] = matrix[:3, 3]
                self.fk_rb_quat[i] = transforms.matrix_to_quaternion(matrix[:3, :3])
            self.fk_rb_pos[:] = transforms.quaternion_apply(self.root_quat[None, :], self.fk_rb_pos) + self.root_pos[None, :]
            self.fk_rb_quat[:] = transforms.quaternion_multiply(self.root_quat[None, :], self.fk_rb_quat)

        else:
            self.mj_joint_pos[:3] = self.root_pos.numpy()
            self.mj_joint_pos[3:7] = self.root_quat.numpy()
            self.mj_joint_pos[self.mj_joint_ids] = joint_pos.numpy()
            self.mj_data.qpos[:] = self.mj_joint_pos
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.mj_viewer.sync()

            # use mujoco for fk computation
            self.fk_rb_pos[:] = torch.from_numpy(self.mj_data.xpos[self.mj_body_ids].copy()).float()
            self.fk_rb_quat[:] = torch.from_numpy(self.mj_data.xquat[self.mj_body_ids].copy()).float()

def start_fk_service(urdf_path: str, 
                     joint_names: list[str],
                     body_names: list[str], 
                     qpos_shm_name: str,
                     body_pose_shm_name: str,
                     debug_spheres_shm_name: str | None = None,
                     xml_path: str | None = None,
                     enable_viewer: bool = False,
                     max_viewer_spheres: int = 0,
                     max_freq: float = 500.0):
    fk_service = FKService(urdf_path, joint_names, body_names, xml_path, enable_viewer=enable_viewer, max_viewer_spheres=max_viewer_spheres)
    qpos_shm = SharedMemory(name=qpos_shm_name)
    body_pose_shm = SharedMemory(name=body_pose_shm_name)
    qpos_shm_array = np.ndarray(shape=(7+len(joint_names),), dtype=np.float32, buffer=qpos_shm.buf)
    body_pose_shm_array = np.ndarray(shape=(len(body_names), 7), dtype=np.float32, buffer=body_pose_shm.buf)

    if max_viewer_spheres > 0:
        assert debug_spheres_shm_name is not None, "debug_spheres_shm_name must be provided when max_viewer_spheres > 0"
        debug_spheres_shm = SharedMemory(name=debug_spheres_shm_name)
        debug_spheres_shm_array = np.ndarray(shape=(max_viewer_spheres, 7), dtype=np.float32, buffer=debug_spheres_shm.buf)
    else:
        debug_spheres_shm_array = None

    qpos = torch.from_numpy(qpos_shm_array)
    body_pose = torch.from_numpy(body_pose_shm_array)

    while True:
        start_time = time.monotonic()
        if debug_spheres_shm_array is not None:
            fk_service.update_debug_spheres(debug_spheres_shm_array[:, :3], debug_spheres_shm_array[:, 3:7])

        fk_service.update_root_state(qpos[:3], qpos[3:7])
        fk_service.forward_kinematics(qpos[7:])
        body_pose[:] = torch.cat([fk_service.fk_rb_pos, fk_service.fk_rb_quat], dim=-1)
        end_time = time.monotonic()

        delta_time = end_time - start_time
        if delta_time < 1.0 / max_freq:
            time.sleep(max(0, 1.0 / max_freq - delta_time - 0.001))

if __name__ == "__main__":
    joint_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
    body_names = ['left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link']
    service = FKService(
        urdf_path="deploy_utils/assets/h1_2.urdf",
        joint_names=joint_names,
        body_names=body_names,
        xml_path="deploy_utils/assets/h1_2.xml",
        enable_viewer=True,
    )
    joint_pos = torch.zeros(len(joint_names))
    joint_pos[0] = 1.0
    service.update_root_state(torch.zeros(3), torch.zeros(4))
    service.forward_kinematics(torch.zeros(len(joint_names)))
    while True:
        service.update_root_state(torch.zeros(3), torch.zeros(4))
        service.forward_kinematics(joint_pos)
        time.sleep(0.01)