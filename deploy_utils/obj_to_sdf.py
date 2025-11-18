import torch
import numpy as np
import trimesh
import pytorch_kinematics.transforms as transforms

class ObjectSDF:
    def __init__(self, points, normals, device='cuda'):
        assert isinstance(points, torch.Tensor) and isinstance(normals, torch.Tensor), "points and normals must be torch tensors"
        self.points = points.float()
        self.normals = normals.float()

        import torch_kdtree
        self.kd_tree = torch_kdtree.build_kd_tree(points, device=device)

    @staticmethod
    def build_from_mesh(mesh_path: str, offset: torch.Tensor | None = None, device='cuda', 
                        scan_count=20, scan_resolution=20, sample_point_count=1000) -> 'ObjectSDF':
        mesh = trimesh.load(mesh_path)

        from mesh_to_sdf import get_surface_point_cloud
        cloud = get_surface_point_cloud(mesh, surface_point_method='scan', 
                                        scan_count=scan_count, scan_resolution=scan_resolution, sample_point_count=sample_point_count)
        return ObjectSDF.build_from_cloud(cloud.points, cloud.normals, offset=offset, device=device)
    
    @staticmethod
    def build_from_cloud(points: np.ndarray, normals: np.ndarray, 
                         offset: torch.Tensor | None = None, device='cuda') -> 'ObjectSDF':
        points = torch.from_numpy(points).to(device).float()
        normals = torch.from_numpy(normals).to(device).float()
        if offset is not None:
            assert offset.shape == (7,), "offset must be a 7D vector"
            points = points + offset[:3]
            normals = transforms.quaternion_apply(offset[3:7], normals)
        return ObjectSDF(points, normals, device)

    @torch.inference_mode()
    def query(self, query_points: torch.Tensor, 
              object_pos: torch.Tensor | None = None,
              object_rot: torch.Tensor | None = None,
              k: int = 5,
              relative_gradient: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(query_points, torch.Tensor), "query_points must be a torch tensor"
        assert query_points.device == self.points.device, "query_points must be on the same device as points"
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

        query_shape = query_points.shape
        query_points = query_points.contiguous().view(-1, 3)

        distances, indices = self.kd_tree.query(query_points, nr_nns_searches=k)
        distances = torch.sqrt(distances)

        closest_points = self.points[indices]
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
    def query_with_normals(self, query_points: torch.Tensor, 
                            object_pos: torch.Tensor | None = None,
                            object_rot: torch.Tensor | None = None,
                            k: int = 5,
                            relative_gradient: bool = False,
                            relative_normals: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(query_points, torch.Tensor), "query_points must be a torch tensor"
        assert query_points.device == self.points.device, "query_points must be on the same device as points"
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

        query_shape = query_points.shape
        query_points = query_points.contiguous().view(-1, 3)

        distances, indices = self.kd_tree.query(query_points, nr_nns_searches=k)
        distances = torch.sqrt(distances)

        closest_points = self.points[indices]
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

        normals = self.normals[indices[:, 0]].view(*query_shape)
        if object_rot is not None and not relative_gradient:
            gradients = transforms.quaternion_apply(object_rot, gradients)
        if object_rot is not None and not relative_normals:
            normals = transforms.quaternion_apply(object_rot, normals)
        return distances, gradients, normals