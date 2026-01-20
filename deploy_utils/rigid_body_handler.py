from tkinter import S
import torch
import torch.multiprocessing as mp
import pickle
import time
import pytorch_kinematics.transforms as transforms
from .obj_to_sdf import ObjectSDF

# Set multiprocessing start method once at module level
mp.set_start_method('spawn', force=True)

@torch.inference_mode()
def _async_query_closest_sdf_worker(shared_state, shared_rb_datas, shared_rb_offsets, 
                                     shared_query_points, output_queue, 
                                     frequency, max_valid_distance, weighted_gradient, 
                                     weight_temperature, add_ground_sdf):
    """Standalone worker function for async SDF queries in a separate process.
    
    Args:
        shared_state: Dictionary with device and rb_sdfs (non-updatable)
        shared_rb_datas: Dict of shared tensors for rigid body data (updatable)
        shared_rb_offsets: Dict of shared tensors for rigid body offsets (updatable)
        shared_query_points: Shared tensor for query points (updatable)
    """
    import torch
    import time
    import pytorch_kinematics.transforms as transforms
    
    device = shared_state['device']
    rb_sdfs = shared_state['rb_sdfs']
    
    # Create views of shared tensors on the target device
    # These will be updated by the main process
    query_points = shared_query_points.to(device)
    
    dt = 1.0 / frequency
    stream = torch.cuda.Stream(device=device)
    
    def query_closest_sdf(query_points, max_valid_distance, weighted_gradient, 
                          weight_temperature, add_ground_sdf):
        """Local query function that reads from shared memory."""
        distances = []
        gradients = []
        valid_masks = []
        
        if add_ground_sdf:
            dist = query_points[..., 2].clone().clamp(min=0.0)
            grad = torch.zeros_like(query_points)
            grad[..., 2] = 1.0
            valid_mask = dist < max_valid_distance
            distances.append(dist * valid_mask + max_valid_distance * (~valid_mask))
            gradients.append(grad * valid_mask.unsqueeze(-1))
            valid_masks.append(valid_mask)

        for k, sdf in rb_sdfs.items():
            if k not in shared_rb_datas:
                continue
            # Read latest data from shared memory (always use the last element)
            object_pos, object_rot = shared_rb_datas[k][-1:, :3].to(device), shared_rb_datas[k][-1:, 3:].to(device)
            if k in shared_rb_offsets:
                object_pos = object_pos + shared_rb_offsets[k][-1:].to(device)
            dist, grad = sdf.query(query_points, object_pos, object_rot)
            valid_mask = dist < max_valid_distance
            distances.append(dist * valid_mask + max_valid_distance * (~valid_mask))
            gradients.append(grad * valid_mask.unsqueeze(-1))
            valid_masks.append(valid_mask)
        
        if len(distances) == 0:
            # No valid SDFs, return dummy values
            return torch.full((query_points.shape[0],), max_valid_distance, device=device), \
                   torch.zeros_like(query_points)
        
        distances = torch.stack(distances, dim=1)
        gradients = torch.stack(gradients, dim=1)
        valid_masks = torch.stack(valid_masks, dim=1)

        min_dist = distances.min(dim=1)
        min_dist, dist_indices = min_dist.values, min_dist.indices
        if not weighted_gradient:
            point_indices = torch.arange(query_points.shape[0], device=device)
            gradients = gradients[point_indices, dist_indices]
        else:
            weight = torch.softmax((max_valid_distance - distances) / (max_valid_distance * weight_temperature), dim=1)
            weight = weight.masked_fill(~valid_masks, float('-inf'))
            weight = torch.nan_to_num(weight, nan=0.0)
            gradients = (gradients * weight.unsqueeze(-1)).sum(dim=1)
        return min_dist, gradients
    
    with torch.cuda.stream(stream):
        while True:
            try:
                start_time = time.monotonic()
                # Read latest query points from shared memory
                current_query_points = shared_query_points.to(device)
                dist, grad = query_closest_sdf(
                    query_points=current_query_points,
                    max_valid_distance=max_valid_distance,
                    weighted_gradient=weighted_gradient,
                    weight_temperature=weight_temperature,
                    add_ground_sdf=add_ground_sdf
                )
                elapsed_time = time.monotonic() - start_time
                time_remaining = dt - elapsed_time
                
                # Try to put results into queue - use put_nowait to avoid blocking
                # If queue is full, skip this frame to avoid blocking
                try:
                    output_queue.put_nowait((dist, grad))
                except Exception as e:
                    # Queue is full (queue.Full) or other error - skip this frame
                    # This prevents blocking when consumer is slow
                    pass
                
                # Wait for remaining time to maintain frequency
                # If query took longer than dt, skip the wait to avoid tight loop
                if time_remaining > 0:
                    while time.monotonic() - start_time < dt:
                        time.sleep(0.0001)
                else:
                    # Query took longer than dt - continue immediately
                    pass
            except Exception as e:
                import traceback
                print(f"Error in SDF worker: {e}", flush=True)
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retrying


class RigidBodyHandler:
    def __init__(self, history_length: int = 5, device: str = 'cuda'):
        self.rb_datas: dict[str, torch.Tensor] = {}
        self.rb_offsets: dict[str, torch.Tensor] = {}
        self.rb_sdfs: dict[str, ObjectSDF] = {}

        self.history_length = history_length
        self.device = device
        
        # Shared memory tensors for async SDF queries
        self._shared_rb_datas: dict[str, torch.Tensor] = {}
        self._shared_rb_offsets: dict[str, torch.Tensor] = {}
        self._shared_query_points: torch.Tensor | None = None
        self._sdf_query_process: mp.Process | None = None
        self._sdf_output_queue: mp.Queue | None = None
    
    def __del__(self):
        """Cleanup async query process on destruction."""
        self.stop_async_query()

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
                sdf = ObjectSDF.build_from_mesh(mesh_path, device=self.device)
            elif cloud_path is not None:
                with open(cloud_path, 'rb') as f:
                    cloud = pickle.load(f)
                sdf = ObjectSDF.build_from_cloud(
                    points=cloud['points'], normals=cloud['normals'], device=self.device)
            elif points is not None and normals is not None:
                sdf = ObjectSDF.build_from_cloud(points, normals, device=self.device)
            else:
                sdf = None
            if sdf is not None:
                self.rb_sdfs[name] = sdf
        
        assert data.shape[0] == 7, "data must be a 7D tensor, [pos, quat]"
        if name not in self.rb_datas:
            self.rb_datas[name] = torch.zeros(self.history_length, 7, device=self.device)
        self.rb_datas[name][:] = self.rb_datas[name].roll(shifts=-1, dims=0)
        self.rb_datas[name][-1] = data

        if offset is not None:
            if name not in self.rb_offsets:
                self.rb_offsets[name] = torch.zeros(self.history_length, 3, device=self.device)
            self.rb_offsets[name][:] = self.rb_offsets[name].roll(shifts=-1, dims=0)
            self.rb_offsets[name][-1] = offset
        
        # Automatically update shared memory if async query is running
        if self._sdf_query_process is not None and self._sdf_query_process.is_alive():
            self.update_async_rigid_body_data(name)

    @torch.inference_mode()
    def query_closest_sdf(self, query_points: torch.Tensor,
                            max_valid_distance: float = 1.0,
                            weighted_gradient: bool = True,
                            weight_temperature: float = 0.5,
                            add_ground_sdf: bool = False):
        assert max_valid_distance > 0.0, "max_valid_distance must be greater than 0.0"

        distances = []
        gradients = []
        valid_masks = []
        if add_ground_sdf:
            dist = query_points[..., 2].clone().clamp(min=0.0)
            grad = torch.zeros_like(query_points)
            grad[..., 2] = 1.0
            valid_mask = dist < max_valid_distance
            distances.append(dist * valid_mask + max_valid_distance * (~valid_mask))
            gradients.append(grad * valid_mask.unsqueeze(-1))
            valid_masks.append(valid_mask)

        for k, sdf in self.rb_sdfs.items():
            object_pos, object_rot = self.rb_datas[k][-1:, :3], self.rb_datas[k][-1:, 3:]
            if k in self.rb_offsets:
                object_pos = object_pos + self.rb_offsets[k][-1:]
            dist, grad = sdf.query(query_points, object_pos, object_rot)
            valid_mask = dist < max_valid_distance
            distances.append(dist * valid_mask + max_valid_distance * (~valid_mask))
            gradients.append(grad * valid_mask.unsqueeze(-1))
            valid_masks.append(valid_mask)
        
        distances = torch.stack(distances, dim=1)
        gradients = torch.stack(gradients, dim=1)
        valid_masks = torch.stack(valid_masks, dim=1)

        min_dist = distances.min(dim=1)
        min_dist, dist_indices = min_dist.values, min_dist.indices
        if not weighted_gradient:
            point_indices = torch.arange(query_points.shape[0], device=self.device)
            gradients = gradients[point_indices, dist_indices]
        else:
            weight = torch.softmax((max_valid_distance - distances) / (max_valid_distance * weight_temperature), dim=1)
            weight = weight.masked_fill(~valid_masks, float('-inf'))
            weight = torch.nan_to_num(weight, nan=0.0)
            gradients = (gradients * weight.unsqueeze(-1)).sum(dim=1)
        return min_dist, gradients

    @torch.inference_mode()
    def async_query_closest_sdf(self, 
                                query_points: torch.Tensor,
                                frequency: float = 50.0,
                                max_valid_distance: float = 1.0,
                                weighted_gradient: bool = True,
                                weight_temperature: float = 0.5,
                                add_ground_sdf: bool = False):
        """Start async SDF query process with shared memory for dynamic updates."""
        if self._sdf_query_process is not None and self._sdf_query_process.is_alive():
            print("SDF query process already running, stopping it first", flush=True)
            self.stop_async_query()
        
        output_queue = mp.Queue(maxsize=1)
        print("Starting SDF query process", flush=True)
        
        # Create shared memory tensors for rb_datas
        shared_rb_datas = {}
        for k, v in self.rb_datas.items():
            # Create shared tensor on CPU (will be moved to device in worker)
            shared_tensor = torch.zeros_like(v.cpu())
            shared_tensor.share_memory_()
            shared_tensor.copy_(v.cpu())
            shared_rb_datas[k] = shared_tensor
        
        # Create shared memory tensors for rb_offsets
        shared_rb_offsets = {}
        if self.rb_offsets:
            for k, v in self.rb_offsets.items():
                shared_tensor = torch.zeros_like(v.cpu())
                shared_tensor.share_memory_()
                shared_tensor.copy_(v.cpu())
                shared_rb_offsets[k] = shared_tensor
        
        # Create shared memory tensor for query_points
        shared_query_points = torch.zeros_like(query_points.cpu())
        shared_query_points.share_memory_()
        shared_query_points.copy_(query_points.cpu())
        
        # Store references for updates
        self._shared_rb_datas = shared_rb_datas
        self._shared_rb_offsets = shared_rb_offsets
        self._shared_query_points = shared_query_points
        self._sdf_output_queue = output_queue
        
        # Create a shared state dictionary that can be passed to the process
        # Note: ObjectSDF objects must be picklable for this to work
        try:
            shared_state = {
                'device': self.device,
                'rb_sdfs': self.rb_sdfs,  # These must be picklable
            }
            
            # Test if the state can be pickled before creating the process
            import pickle
            pickle.dumps(shared_state)
        except Exception as e:
            print(f"ERROR: Cannot pickle shared state for SDF worker process: {e}", flush=True)
            print("This usually means ObjectSDF objects contain non-picklable data (e.g., CUDA tensors)", flush=True)
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot create SDF worker process: objects are not picklable: {e}")
        
        self._sdf_query_process = mp.Process(
            target=_async_query_closest_sdf_worker, 
            args=(
                shared_state,
                shared_rb_datas,
                shared_rb_offsets,
                shared_query_points,
                output_queue, 
                frequency, 
                max_valid_distance, 
                weighted_gradient, 
                weight_temperature, 
                add_ground_sdf
            )
        )
        self._sdf_query_process.daemon = True  # Ensure process dies when parent dies
        try:
            self._sdf_query_process.start()
            # Give the process a moment to start and check if it's alive
            time.sleep(0.1)
            if not self._sdf_query_process.is_alive():
                exitcode = self._sdf_query_process.exitcode
                raise RuntimeError(f"SDF query process died immediately with exit code: {exitcode}")
            print(f"SDF query process started successfully, PID: {self._sdf_query_process.pid}", flush=True)
        except Exception as e:
            print(f"Failed to start SDF query process: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        return self._shared_query_points, output_queue
    
    @torch.inference_mode()
    def update_async_query_points(self, query_points: torch.Tensor):
        """Update query points in shared memory for async SDF queries."""
        if self._shared_query_points is None:
            raise RuntimeError("Async SDF query not started. Call async_query_closest_sdf first.")
        if query_points.shape != self._shared_query_points.shape:
            raise ValueError(f"Query points shape mismatch: got {query_points.shape}, expected {self._shared_query_points.shape}")
        self._shared_query_points.copy_(query_points.cpu())
    
    @torch.inference_mode()
    def update_async_rigid_body_data(self, name: str):
        """Update rigid body data in shared memory for async SDF queries."""
        if name not in self._shared_rb_datas:
            # Create new shared tensor if it doesn't exist
            if name not in self.rb_datas:
                raise ValueError(f"Rigid body '{name}' not found in rb_datas")
            shared_tensor = torch.zeros_like(self.rb_datas[name].cpu())
            shared_tensor.share_memory_()
            self._shared_rb_datas[name] = shared_tensor
        
        self._shared_rb_datas[name].copy_(self.rb_datas[name].cpu())
        
        # Also update offsets if they exist
        if name in self.rb_offsets:
            if name not in self._shared_rb_offsets:
                shared_tensor = torch.zeros_like(self.rb_offsets[name].cpu())
                shared_tensor.share_memory_()
                self._shared_rb_offsets[name] = shared_tensor
            self._shared_rb_offsets[name].copy_(self.rb_offsets[name].cpu())
    
    def stop_async_query(self):
        """Stop the async SDF query process."""
        if self._sdf_query_process is not None and self._sdf_query_process.is_alive():
            self._sdf_query_process.terminate()
            self._sdf_query_process.join(timeout=1.0)
            if self._sdf_query_process.is_alive():
                self._sdf_query_process.kill()
                self._sdf_query_process.join()
            print("SDF query process stopped", flush=True)
        self._sdf_query_process = None
        self._shared_rb_datas = {}
        self._shared_rb_offsets = {}
        self._shared_query_points = None
        self._sdf_output_queue = None