import torch
import numpy as np
import pytorch_kinematics.transforms as transforms

import os
import sys
from copy import deepcopy
import trimesh

import importlib

try:
    importlib.util.find_spec('kaolin')
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from kaolin.ops.mesh import check_sign, index_vertices_by_faces, face_normals
    _KAOLIN_AVAILABLE = True
except ImportError:
    _KAOLIN_AVAILABLE = False

class MeshModel:
    @torch.no_grad()
    @torch.inference_mode(False)
    def __init__(self, mesh_path=None, device='cuda', **kwargs):
        self.device = device

        self.mesh_path = mesh_path
        mesh = trimesh.load_mesh(self.mesh_path, force='mesh')
        self.mesh = mesh
        
        # Note that for parallel computation in kaolin, the mesh is loaded as 1-batch.
        # 3D computations are carried out in the object-centered frame.
        self.canon_object_verts = torch.Tensor(mesh.vertices).to(self.device).unsqueeze(0)
        self.object_faces = torch.Tensor(mesh.faces).long().to(self.device)
        self.object_face_verts = index_vertices_by_faces(self.canon_object_verts, self.object_faces)
        self.object_verts = self.canon_object_verts.clone()
        self.object_face_normals = face_normals(self.object_face_verts, unit=True)

    def distance(self, points, signed=False):
        B, N, _ = points.shape
        points = points.reshape([1, -1, 3]).contiguous()
        dis_sq, _, _ = point_to_mesh_distance(points, self.object_face_verts)
        dis = torch.sqrt(dis_sq)
        if signed:
            signs = check_sign(self.object_verts, self.object_faces, points)
            dis = torch.where(signs, -dis, dis)
        return dis.reshape([B, N])

    def gradient(self, points, distance):
        grad = torch.autograd.grad([distance.sum()], [points], create_graph=True, allow_unused=True)[0]
        return grad / grad.norm(dim=-1, keepdim=True)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def distance_gradient(self, points, signed=True):
        point_shape = points.shape[:-1]
        
        # Ensure points have requires_grad=True for gradient computation
        # If points doesn't have requires_grad, clone and set it
        if not points.requires_grad:
            points = points.clone().detach().requires_grad_(True)
        
        points = points.reshape(1, -1, 3).contiguous()
        dis, face_idx, _ = point_to_mesh_distance(points, self.object_face_verts)
        dis = torch.sqrt(dis)

        if signed:
            signs = check_sign(self.object_verts, self.object_faces, points)
            dis = torch.where(signs, -dis, dis)

        gradient = torch.autograd.grad([dis.sum()], [points], allow_unused=False, create_graph=False, retain_graph=False)[0]
        gradient = gradient / (torch.norm(gradient, dim=-1, keepdim=True) + 1e-12)

        return dis.reshape(*point_shape), gradient.reshape(*point_shape, 3)

    def get_obj_mesh(self):
        return deepcopy(self.mesh)

class ObjectSDF:
    def __init__(self, 
                 mesh_model: MeshModel | None = None,
                 points: torch.Tensor | None = None, 
                 normals: torch.Tensor | None = None, 
                 device='cuda', 
                 backend='native'):
        self.backend = backend
        if self.backend == 'kaolin' and not _KAOLIN_AVAILABLE:
            print("[WARNING]: Kaolin is not available. Using native backend instead.")
            self.backend = 'native'

        if self.backend == 'native':
            assert isinstance(points, torch.Tensor) and isinstance(normals, torch.Tensor), "points and normals must be torch tensors"
            self.points = points.float()
            self.normals = normals.float()
            self.device = points.device

            import torch_kdtree
            self.kd_tree = torch_kdtree.build_kd_tree(points, device=device)
        elif self.backend == 'kaolin':
            assert mesh_model is not None, "mesh_model must be provided if backend is kaolin"
            self.mesh_model = mesh_model
            self.points = mesh_model.object_verts[0]
            self.device = mesh_model.object_faces.device
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    @staticmethod
    def build_from_mesh(mesh_path: str, device='cuda', 
                        scan_count=20, scan_resolution=20, sample_point_count=1000) -> 'ObjectSDF':
        if not _KAOLIN_AVAILABLE:
            print("[WARNING]: Kaolin is not available. Using native backend instead.")
            mesh = trimesh.load(mesh_path)

            from mesh_to_sdf import get_surface_point_cloud
            cloud = get_surface_point_cloud(mesh, surface_point_method='scan', 
                                            scan_count=scan_count, scan_resolution=scan_resolution, sample_point_count=sample_point_count)
            return ObjectSDF.build_from_cloud(cloud.points, cloud.normals, device)
        else:
            print("[INFO]: Kaolin is available. Using kaolin backend instead.")
            mesh_model = MeshModel(mesh_path, device=device)
            return ObjectSDF(mesh_model=mesh_model, device=device, backend='kaolin')
    
    @staticmethod
    def build_from_cloud(points: np.ndarray, normals: np.ndarray, device='cuda') -> 'ObjectSDF':
        points = torch.from_numpy(points).to(device).float()
        normals = torch.from_numpy(normals).to(device).float()
        return ObjectSDF(points=points, normals=normals, device=device, backend='native')
    
    @torch.inference_mode()
    def _transform_query_points(self, query_points: torch.Tensor, 
                                object_pos: torch.Tensor | None = None,
                                object_rot: torch.Tensor | None = None,
                                object_scale: torch.Tensor | None = None) -> torch.Tensor:
        assert isinstance(query_points, torch.Tensor), "query_points must be a torch tensor"
        assert query_points.device == self.device, "query_points must be on the same device as points"
        assert query_points.dtype == torch.float32, "query_points must be float32"
        assert query_points.shape[-1] == 3, "The last dimension of query_points must be 3"
        # transform query points to object frame
        if object_pos is not None:
            assert object_pos.ndim == query_points.ndim, "object_pos must have the same dimensions as query_points"
            query_points = query_points - object_pos
        if object_rot is not None:
            assert object_rot.ndim == query_points.ndim, "object_rot must have the same dimensions as query_points"
            object_rot_inv = transforms.quaternion_invert(object_rot)
            query_points = transforms.quaternion_apply(object_rot_inv, query_points)
        if object_scale is not None:
            assert object_pos is not None and object_rot is not None, "object_pos and object_rot must be provided if object_scale is provided"
            assert object_scale.ndim == query_points.ndim, "object_scale must have the same dimensions as query_points"
            object_scale = object_scale.expand_as(query_points).contiguous()
        else:
            object_scale = 1.0

        query_points = query_points / object_scale
        return query_points, object_scale

    @torch.inference_mode()
    def _query_native(self, query_points: torch.Tensor, 
                        object_pos: torch.Tensor | None = None,
                        object_rot: torch.Tensor | None = None,
                        object_scale: torch.Tensor | None = None,
                        k: int = 5,
                        relative_gradient: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        query_shape = query_points.shape
        query_points, object_scale = self._transform_query_points(query_points, object_pos, object_rot, object_scale)
        query_points = query_points.view(-1, 3)

        distances, indices = self.kd_tree.query(query_points, nr_nns_searches=k)
        distances = torch.sqrt(distances)
        if isinstance(object_scale, torch.Tensor):
            object_scale = object_scale.view(-1, 3)
            query_points = query_points * object_scale

        closest_points = self.points[indices]
        if isinstance(object_scale, torch.Tensor):
            closest_points = closest_points * object_scale.unsqueeze(1)
            distances = (closest_points[:, 0] - query_points).norm(dim=-1).unsqueeze(1)

        direction_from_surface = query_points[:, None, :] - closest_points
        inside = torch.einsum('ijk,ijk->ij', direction_from_surface, self.normals[indices]) < 0
        inside = torch.sum(inside, dim=1) > 5 * 0.5
        distances = distances[:, 0]
        distances[inside] *= 0.0

        gradients = direction_from_surface[:, 0]
        near_surface = (torch.abs(distances) < np.sqrt(0.0025**2 * 3) * 3) | inside # 3D 2-norm stdev * 3
        gradients = torch.where(near_surface[:, None], self.normals[indices[:, 0]], gradients)
        gradients /= torch.norm(gradients, dim=1, keepdim=True)

        distances = distances.view(*query_shape[:-1])
        gradients = gradients.view(*query_shape)

        if object_rot is not None and not relative_gradient:
            gradients = transforms.quaternion_apply(object_rot, gradients)
        return distances, gradients
    
    @torch.inference_mode()
    def _query_kaolin(self, query_points: torch.Tensor, 
                        object_pos: torch.Tensor | None = None,
                        object_rot: torch.Tensor | None = None,
                        object_scale: torch.Tensor | None = None,
                        relative_gradient: bool = False,
                        **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        query_points, object_scale = self._transform_query_points(query_points, object_pos, object_rot, object_scale)
        distances, gradients = self.mesh_model.distance_gradient(query_points, signed=True)
        if isinstance(object_scale, torch.Tensor):
            compressed_gradients = gradients * object_scale
            compressed_norm = compressed_gradients.norm(dim=-1)
            distances = distances * compressed_norm
            gradients = compressed_gradients / (compressed_norm.unsqueeze(-1) + 1e-12)
        if object_rot is not None and not relative_gradient:
            gradients = transforms.quaternion_apply(object_rot, gradients)
        return distances, gradients
    
    def query(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == 'native':
            return self._query_native(*args, **kwargs)
        elif self.backend == 'kaolin':
            return self._query_kaolin(*args, **kwargs)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
        