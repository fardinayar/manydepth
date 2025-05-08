# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, Union

import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils

from nerfstudio.configs import method_configs


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "SO3xR3"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: Union[Tuple, float] = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            # --- modification 1
            # self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
            self.step_counter = 0
            
            # self.camera_count = (num_cameras//self.sequence_length) - 1
            # self.sequence_length = 2
            self.camera_count = 6
            self.sequence_length = num_cameras // (self.camera_count + 1)
            
            self.ext_adjustment = torch.nn.Parameter(torch.zeros((self.camera_count, 6), device=device))
            self.lidar_pose_adjs = torch.nn.Parameter(torch.zeros((self.sequence_length, 6), device=device))
            self.lidar_pose_adjs.requires_grad_(False)
        else:
            assert_never(self.config.mode)

    # --- modification 2
    # def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
    #     """Get the pose adjustment."""
    #     return self.pose_adjustment
    
    def _get_ext_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the extrinsic adjustment."""
        return self.ext_adjustment
    
    def _get_lidar_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the poses adjustment of the lidar for each sequence."""
        return self.lidar_pose_adjs

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []
        

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            # --- modification 3
            # outputs.append(exp_map_SO3xR3(self._get_pose_adjustment()[indices, :]))
            # if self.step_counter == 0:
            #     self.non_trainable_camera_indices = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long)
            self.step_counter += 1
            lidar_adjustments = self._get_lidar_adjustment().to(torch.half)
            ext_adjustments = self._get_ext_adjustment().to(torch.half)
            row_to_add = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.half, device="cuda")
                      
            # Create boolean masks for the two cases.
            mask_ext = indices < (self.sequence_length * self.camera_count)
            mask_lidar = ~mask_ext
            
            num_indices = len(indices)
            output = torch.zeros((num_indices, 3, 4), dtype=torch.half, device="cuda")
            
            # --- Process the lidar-only case
            if mask_lidar.any():
                # For these indices, we subtract camera count * sequence length
                lidar_idx = indices[mask_lidar] - (self.sequence_length * self.camera_count)
                # Gather the corresponding adjustments (assumes lidar_adjustments is indexable with a tensor)
                lidar_inputs = lidar_adjustments[lidar_idx]
                # Compute the mapping in batch
                mapped_lidar = exp_map_SO3xR3(lidar_inputs)  # Expected shape: (B, 3, 4)
                output[mask_lidar] = mapped_lidar
            
            # --- Process camera indices in batch ---
            if mask_ext.any():
                # Compute indices for each adjustment type
                lidar_indices = indices[mask_ext] // self.camera_count
                ext_indices = indices[mask_ext] // self.sequence_length
                
                # Gather adjustments for each branch
                lidar_inputs = lidar_adjustments[lidar_indices]
                ext_inputs = ext_adjustments[ext_indices]
                
                # Compute the mappings in batch
                # lidar_pose = exp_map_SO3xR3(lidar_inputs)  # Expected shape: (B, 3, 4)
                ext_adj   = exp_map_SO3xR3(ext_inputs)       # Expected shape: (B, 3, 4)
                
                # # Expand row_to_add for each batch element
                # batch_size = lidar_pose.shape[0]
                # row_to_add_exp = row_to_add.expand(batch_size, 1, 4)
                
                # # Form homogeneous matrices (convert 3x4 to 4x4 by appending the row)
                # lidar_hom = torch.cat((lidar_pose, row_to_add_exp), dim=1)  # (B, 4, 4)
                # ext_hom   = torch.cat((ext_adj, row_to_add_exp), dim=1)      # (B, 4, 4)
                
                # # Multiply the matrices in batch and drop the last row
                # # camera_pose = (ext_hom @ lidar_hom)[:, :-1, :]  # (B, 3, 4)
                # camera_pose = (lidar_hom @ ext_hom)[:, :-1, :]
                
                camera_pose = ext_adj
                
                output[mask_ext] = camera_pose
                    
            outputs.append(output)
            
            # if self.step_counter == method_configs.ITERATION_COUNT - 1:
            if self.step_counter % 2000 == 0 or self.step_counter == 1:
                import json
                # Convert tensor to a JSON-compatible format
                ext_adjustment_list = self.ext_adjustment.detach().cpu().tolist()
                ext_matrix_list = []
                for camera_idx in range(self.camera_count):
                    mapped = exp_map_SO3xR3(self.ext_adjustment[camera_idx].unsqueeze(0))
                    ext_matrix_list.append(mapped.squeeze(0).detach().cpu().tolist())
                
                # Save to a JSON file
                noise_type = "translation"
                camera = "back_camera"
                axis = "x"
                noise_amount = "0.04"
                path = f"calib_results/compilation/{noise_type}/{noise_amount}/{camera}/{axis}"
                # path = f"calib_results/compilation/no_noise_reg"
                with open(f"{path}/ext_adjustment{self.step_counter}.json", "w") as f:
                    json.dump(ext_adjustment_list, f)
                    
                with open(f"{path}/ext_adjustment_matrix{self.step_counter}.json", "w") as f:
                    json.dump(ext_matrix_list, f)
                    
        elif self.config.mode == "SE3":
            # outputs.append(exp_map_SE3(self._get_pose_adjustment()[indices, :]))
            pass
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.ext_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.ext_adjustment.device)
            identity = torch.eye(4, device=self.ext_adjustment.device, dtype=outputs[0].dtype)[:3, :4]
            outputs[0][self.non_trainable_camera_indices] = identity
            
        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = (
                torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None])
                .squeeze()
                .to(raybundle.origins)
            )

    def apply_to_camera(self, camera: Cameras) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            assert camera.metadata is not None, "Must provide id of camera in its metadata"
            assert "cam_idx" in camera.metadata, "Must provide id of camera in its metadata"
            camera_idx = camera.metadata["cam_idx"]
            adj = self(torch.tensor([camera_idx], dtype=torch.long, device=camera.device))  # type: ignore
            adj = torch.cat([adj, torch.Tensor([0, 0, 0, 1])[None, None].to(adj)], dim=1)
            camera.camera_to_worlds = torch.bmm(camera.camera_to_worlds, adj)

    # --- modification 4
    def get_loss_dict_by_sensor_type(self, loss_dict: dict, is_ext: bool) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_ext_adjustment() if is_ext else self._get_lidar_adjustment()
            loss_dict["camera_opt_regularizer"] = (
                pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
                + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )
            
    def get_loss_dict(self, loss_dict: dict) -> None:
        self.get_loss_dict_by_sensor_type(loss_dict, True)
        self.get_loss_dict_by_sensor_type(loss_dict, False)
        

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            # --- modification 5
            # pose_adjustment = self._get_pose_adjustment()
            ext_adjustment = self._get_ext_adjustment()
            metrics_dict["camera_opt_translation"] = ext_adjustment[:, :3].norm()
            metrics_dict["camera_opt_rotation"] = ext_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0


@dataclass
class ScaledCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of axis-masked optimization for camera poses."""

    _target: Type = field(default_factory=lambda: ScaledCameraOptimizer)

    weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    trans_l2_penalty: Union[Tuple[float, float, float], float] = (
        1e-2,
        1e-2,
        1e-2,
    )  # TODO: this is l1


class ScaledCameraOptimizer(CameraOptimizer):
    """Camera optimizer that masks which components can be optimized."""

    def __init__(self, config: ScaledCameraOptimizerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config: ScaledCameraOptimizerConfig = self.config
        self.register_buffer("weights", torch.tensor(self.config.weights, dtype=torch.float32))
        self.trans_penalty = torch.tensor(self.config.trans_l2_penalty, dtype=torch.float32, device=self.device)

    # --- modification 6
    # def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
    #     """Get the pose adjustment."""
    #     return self.pose_adjustment * self.weights
    
    def _get_ext_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.ext_adjustment * self.weights
    
    def _get_lidar_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.lidar_pose_adjs * self.weights

    def get_loss_dict_by_sensor_type(self, loss_dict: dict, is_ext: bool) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_ext_adjustment() if is_ext else self._get_lidar_adjustment()
            self.trans_penalty = self.trans_penalty.to(pose_adjustment.device)
            loss_dict["camera_opt_regularizer"] = (
                pose_adjustment[:, :3].abs() * self.trans_penalty
            ).mean() + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
    
    def get_loss_dict(self, loss_dict: dict) -> None:
        self.get_loss_dict_by_sensor_type(loss_dict, True)
        self.get_loss_dict_by_sensor_type(loss_dict, False)