from tkinter import S
import torch
import pickle
try:
    import torch_kdtree
    import mesh_to_sdf
except ImportError:
    print('[Error]: torch_kdtree and mesh_to_sdf are not installed, RigidBodyHandler will be unavailable')
    import traceback
    traceback.print_exc()
    raise ImportError('torch_kdtree and mesh_to_sdf are not installed')
from .obj_to_sdf import ObjectSDF

class RigidBodyHandler:
    def __init__(self, history_length: int = 5, device: str = 'cuda'):
        self.rb_datas: dict[str, torch.Tensor] = {}
        self.rb_sdfs: dict[str, ObjectSDF] = {}

        self.history_length = history_length
        self.device = device

    @torch.inference_mode()
    def update_rigid_body_data(self, name: str, data: torch.Tensor,
                                offset: torch.Tensor | None = None,
                                mesh_path: str | None = None,
                                cloud_path: str | None = None,
                                points: torch.Tensor | None = None,
                                normals: torch.Tensor | None = None,
                                overwrite_sdf: bool = False):
        if name not in self.rb_sdfs or (name in self.rb_sdfs and overwrite_sdf):
            if mesh_path is not None:
                sdf = ObjectSDF.build_from_mesh(mesh_path, offset=offset, device=self.device)
            elif cloud_path is not None:
                with open(cloud_path, 'rb') as f:
                    cloud = pickle.load(f)
                sdf = ObjectSDF.build_from_cloud(
                    points=cloud['points'], normals=cloud['normals'], offset=offset, device=self.device)
            elif points is not None and normals is not None:
                sdf = ObjectSDF.build_from_cloud(points, normals, offset=offset, device=self.device)
            else:
                sdf = None
            if sdf is not None:
                self.rb_sdfs[name] = sdf
        
        assert data.shape[0] == 7, "data must be a 7D tensor, [pos, quat]"
        if name not in self.rb_datas:
            self.rb_datas[name] = torch.zeros(self.history_length, 7, device=self.device)
        self.rb_datas[name] = self.rb_datas[name].roll(shifts=-1, dims=0)
        self.rb_datas[name][-1] = data

    @torch.inference_mode()
    def query_closest_weighted_sdf(self, query_points: torch.Tensor, max_valid_distance: float | None = None) -> torch.Tensor:
        distances = []
        gradients = []
        valid_masks = []
        for name, sdf in self.rb_sdfs.items():
            object_pos = self.rb_datas[name][-1:, :3]
            object_quat = self.rb_datas[name][-1:, 3:]
            dist, grad = sdf.query(query_points, object_pos, object_quat)
            
            if max_valid_distance is not None:
                valid_mask = dist < max_valid_distance
            else:
                max_valid_distance = 1.0
                valid_mask = torch.ones_like(dist, dtype=torch.bool, device=self.device)
            distances.append(dist * valid_mask + max_valid_distance * (~valid_mask))
            gradients.append(grad * valid_mask.unsqueeze(-1))
            valid_masks.append(valid_mask)

        distances = torch.stack(distances, dim=0)
        gradients = torch.stack(gradients, dim=0)
        valid_masks = torch.stack(valid_masks, dim=0)

        min_dist = distances.min(dim=0).values.clamp(min=0.0)
        weight = distances.masked_fill(~valid_masks, float('-inf'))
        weight = torch.softmax(weight, dim=0)
        weight = torch.nan_to_num(weight, nan=0.0)
        gradients = (gradients * weight.unsqueeze(-1)).sum(dim=0)
        return min_dist, gradients