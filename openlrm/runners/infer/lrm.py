# Copyright (c) 2025 Sam Bahrami
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

import torch
import os
import argparse
import numpy as np
import json
import torchvision
from omegaconf import OmegaConf, DictConfig
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .base_inferrer import Inferrer
from openlrm.utils.logging import configure_logger
from openlrm.runners import REGISTRY_RUNNERS
from openlrm.utils.hf_hub import wrap_model_hub

logger = get_logger(__name__)


def parse_configs() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to training config file')
    parser.add_argument('--infer', type=str, required=True, help='Path to inference config file')
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # Parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    # Load inference config (required)
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
    else:
        raise ValueError("--infer argument is required for evaluation")

    # Optionally load training config for additional parameters
    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        # Only override if not already set in inference config
        cfg.setdefault('source_size', cfg_train.dataset.source_image_res)
        cfg.setdefault('render_size', cfg_train.dataset.render_image.high)

    cfg.merge_with(cli_cfg)

    # Set defaults
    cfg.setdefault('logger', 'INFO')
    cfg.setdefault('app_enabled', False)

    # Validate required parameters
    assert cfg.model_name is not None, "model_name is required"
    assert hasattr(cfg, 'dataset'), "dataset configuration is required"
    
    return cfg


@REGISTRY_RUNNERS.register('infer.lrm')
class LRMInferrer(Inferrer):

    EXP_TYPE: str = 'lrm'

    def __init__(self) -> None:
        super().__init__()

        self.cfg = parse_configs()
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )
        self.model = self._build_model(self.cfg).to(self.device)

    def _build_model(self, cfg: DictConfig) -> Any:
        from openlrm.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _load_and_normalize_poses(self, root_dir: str, uid: str, conditioning_views: List[int], 
                                  all_views: List[int], device: torch.device) -> Tuple[torch.Tensor, Dict[int, int]]:
        """Load all poses and normalize them with conditioning view 64 as reference"""
        from openlrm.datasets.cam_utils import camera_normalization_objaverse
        
        pose_dir = os.path.join(root_dir, uid, "pose")
        
        # Load ALL poses with conditioning view 64 first (for proper normalization reference)
        poses_for_normalization = []
        pose_index_mapping = {}  # Maps original view_idx to index in poses_for_normalization
        
        # First, load the primary conditioning view (64) - this will be the normalization reference
        primary_conditioning_view = conditioning_views[0]  # Should be 64
        pose_path = os.path.join(pose_dir, f"{primary_conditioning_view:06d}.txt")
        primary_pose = self._load_pose(pose_path)
        poses_for_normalization.append(primary_pose)
        pose_index_mapping[primary_conditioning_view] = 0
        
        # Then load all other poses (excluding the primary conditioning view we already loaded)
        current_idx = 1
        for view_idx in all_views:
            if view_idx != primary_conditioning_view:
                pose_path = os.path.join(pose_dir, f"{view_idx:06d}.txt")
                pose = self._load_pose(pose_path)
                poses_for_normalization.append(pose)
                pose_index_mapping[view_idx] = current_idx
                current_idx += 1
        
        all_poses = torch.stack(poses_for_normalization, dim=0)
        
        # Normalize all cameras together with conditioning view 64 as reference (index 0)
        if self.cfg.dataset.normalize_camera:
            all_poses = camera_normalization_objaverse(self.cfg.dataset.normed_dist_to_center, all_poses)
        
        # Move poses to device
        all_poses = all_poses.to(device)
        
        return all_poses, pose_index_mapping
    
    def _load_conditioning_data(self, root_dir: str, uid: str, conditioning_views: List[int], 
                                all_poses: torch.Tensor, pose_index_mapping: Dict[int, int], 
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load conditioning images and build source camera"""
        from openlrm.datasets.cam_utils import build_camera_principle
        
        rgba_dir = os.path.join(root_dir, uid, "rgb")
        intrinsics_path = os.path.join(root_dir, uid, "intrinsics.txt")
        
        # Load intrinsics
        intrinsics = self._parse_intrinsics(intrinsics_path, self.cfg.source_size)
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        
        # Build K intrinsics matrix
        fx, fy, cx, cy = intrinsics_tensor[0, 0], intrinsics_tensor[0, 1], intrinsics_tensor[1, 0], intrinsics_tensor[1, 1]
        K_intrinsics_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        
        # Extract conditioning poses from normalized poses using the mapping
        conditioning_poses = []
        for view_idx in conditioning_views:
            pose_idx = pose_index_mapping[view_idx]
            conditioning_poses.append(all_poses[pose_idx])
        conditioning_poses = torch.stack(conditioning_poses, dim=0)
        
        # Load conditioning images
        conditioning_images = []
        for view_idx in conditioning_views:
            rgba_path = os.path.join(rgba_dir, f"{view_idx:06d}.png")
            image = self._load_rgba_image(rgba_path, bg_color=1.0)
            conditioning_images.append(image)
        
        conditioning_images = torch.cat(conditioning_images, dim=0)  # Concatenate along channel dimension
        
        # Build source camera from the primary conditioning view (view 64, now at index 0)
        primary_pose = all_poses[0:1]  # The primary conditioning view is now at index 0
        source_camera = build_camera_principle(
            primary_pose, intrinsics_tensor.unsqueeze(0).to(device)
        ).squeeze(0)
        
        # Adjust source image resolution
        conditioning_images = torch.nn.functional.interpolate(
            conditioning_images.unsqueeze(0),
            size=(self.cfg.source_size, self.cfg.source_size),
            mode="bicubic",
            align_corners=True,
        ).squeeze(0)
        conditioning_images = torch.clamp(conditioning_images, 0, 1)
        
        return conditioning_images, conditioning_poses, source_camera, K_intrinsics_matrix, intrinsics_tensor
    
    def _render_view(self, planes: torch.Tensor, pose: torch.Tensor, gt_image_path: str, 
                     intrinsics_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render a single view and load ground truth image"""
        from openlrm.datasets.cam_utils import build_camera_standard
        
        # Load and process GT image
        gt_image = self._load_rgba_image(gt_image_path, bg_color=1.0)
        gt_image = torch.nn.functional.interpolate(
            gt_image.unsqueeze(0),
            size=(self.cfg.render_size, self.cfg.render_size),
            mode="bicubic",
            align_corners=True,
        ).squeeze(0).to(device)
        gt_image = torch.clamp(gt_image, 0, 1)
        
        # Build render camera
        render_camera = build_camera_standard(
            pose.unsqueeze(0), intrinsics_tensor.unsqueeze(0).to(device)
        ).unsqueeze(0)
        
        # Prepare rendering inputs
        render_anchors = torch.zeros(1, 1, 2, device=device)
        render_resolutions = torch.ones(1, 1, 1, device=device) * self.cfg.render_size
        render_bg_colors = torch.ones(1, 1, 1, device=device)
        
        # Render view
        outs = self.model.synthesizer(
            planes, render_camera, render_anchors, 
            render_resolutions, render_bg_colors, self.cfg.render_size
        )
        rendered_image = outs["images_rgb"][0, 0]
        
        return rendered_image, gt_image
    
    def _infer_triplane_representation(self, conditioning_images: torch.Tensor, source_camera: torch.Tensor, 
                                        K_intrinsics_matrix: torch.Tensor, conditioning_poses: torch.Tensor, 
                                        intrinsics_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Infer triplane representation from conditioning data"""
        from openlrm.datasets.cam_utils import build_camera_standard
        
        if self.model.model_type == "implicit" or self.model.model_type == "explicit":
            # Build render cameras for conditioning views for models that need them
            conditioning_render_cameras = build_camera_standard(
                conditioning_poses, intrinsics_tensor.repeat(conditioning_poses.shape[0], 1, 1).to(device)
            ).unsqueeze(0)
            
            planes, image_feats = self.model.forward_planes(
                conditioning_images.unsqueeze(0).to(device), 
                source_camera.unsqueeze(0).to(device), 
                K_intrinsics_matrix.unsqueeze(0).to(device), 
                conditioning_poses.unsqueeze(0), 
                render_cameras=conditioning_render_cameras
            )
        else:
            planes, image_feats = self.model.forward_planes(
                conditioning_images.unsqueeze(0).to(device), 
                source_camera.unsqueeze(0).to(device)
            )
        
        return planes, image_feats
    
    def infer(self) -> None:


        # Load UIDs directly from validation JSON file
        val_json_path = None
        for subset in self.cfg.dataset.subsets:
            val_json_path = subset.meta_path.val
            root_dir = subset.root_dirs[0]
            break
        
        if val_json_path is None:
            raise ValueError("Could not find metrics_shapenet subset in config")
        
        with open(val_json_path, 'r') as f:
            uids = json.load(f)
        
        # Conditioning views to load (64, 128)
        conditioning_views = [64, 128]
        
        # All views to render (0-250, excluding conditioning views for novel view synthesis)
        all_views = list(range(251))
        novel_views = [v for v in all_views if v not in conditioning_views]
        
        # Get a reasonable name for the model from the cfg.model_name
        model_name = self.cfg.model_name.split("/")[-1]

        # Save to evaluation output directory
        eval_output_dir = Path(self.cfg.model_name) / "evaluation_outputs"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Check for GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Process each UID
        for uid in tqdm(uids):
            print(f"Processing UID: {uid}")
            
            # Create output directories
            uid_output_dir = eval_output_dir / uid
            uid_output_dir.mkdir(parents=True, exist_ok=True)
            uid_gt_dir = Path(str(eval_output_dir) + "_gt") / uid
            uid_gt_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and normalize all poses with conditioning view 64 as reference
            all_poses, pose_index_mapping = self._load_and_normalize_poses(
                root_dir, uid, conditioning_views, all_views, device
            )
            
            # Load conditioning data (images, poses, cameras)
            conditioning_images, conditioning_poses, source_camera, K_intrinsics_matrix, intrinsics_tensor = self._load_conditioning_data(
                root_dir, uid, conditioning_views, all_poses, pose_index_mapping, device
            )

            with torch.no_grad():
                # Infer triplane representation
                planes, image_feats = self._infer_triplane_representation(
                    conditioning_images, source_camera, K_intrinsics_matrix, 
                    conditioning_poses, intrinsics_tensor, device
                )
                
                # Render and save novel views
                gt_images, out_images = [], []
                rgba_dir = os.path.join(root_dir, uid, "rgb")
                
                for i, view_idx in enumerate(novel_views):
                    rgba_path = os.path.join(rgba_dir, f"{view_idx:06d}.png")
                    pose_idx = pose_index_mapping[view_idx]
                    pose = all_poses[pose_idx]
                    
                    rendered_image, gt_image = self._render_view(
                        planes, pose, rgba_path, intrinsics_tensor, device
                    )
                    
                    # Save images
                    filename = f"{i:06d}.png"
                    torchvision.utils.save_image(gt_image, uid_gt_dir / filename)
                    torchvision.utils.save_image(rendered_image, uid_output_dir / filename)
                    
                    # Collect every 32nd image for visualization grid (sample evenly across all views)
                    if i % 32 == 0 and len(gt_images) < 8:
                        gt_images.append(gt_image.unsqueeze(0))
                        out_images.append(rendered_image.unsqueeze(0))
                
                # Create visualization grid
                if gt_images and out_images:
                    gt_images = torch.cat(gt_images, dim=0)
                    out_images = torch.cat(out_images, dim=0)
                    joint_output = torch.cat([gt_images, out_images], dim=0)
                    inputs_and_outputs = torchvision.utils.make_grid(joint_output, nrow=8)
                    filename = f"{model_name}_{uid}.png"
                    torchvision.utils.save_image(inputs_and_outputs, eval_output_dir / filename)

                print(f"UID: {uid} - Images saved successfully")
            
        print("Image generation complete!")
        print(f"- Individual images saved in {eval_output_dir}")
        print(f"- GT images saved in {str(eval_output_dir)}_gt")
        print(f"- Visualization grids saved in {eval_output_dir}")
    
    def _parse_intrinsics(self, filepath: str, trgt_sidelength: Optional[int] = None, invert_y: bool = False) -> np.ndarray:
        """Parse intrinsics from intrinsics.txt file"""
        with open(filepath, "r") as file:
            f, cx, cy, _ = map(float, file.readline().split())
            _ = list(map(float, file.readline().split()))  # grid_barycenter (unused)
            _ = float(file.readline())  # scale (unused)
            height, width = map(float, file.readline().split())

            try:
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        if world2cam_poses is None:
            world2cam_poses = False

        world2cam_poses = bool(world2cam_poses)

        if trgt_sidelength is not None:
            cx = cx / width * trgt_sidelength
            cy = cy / height * trgt_sidelength
            f = trgt_sidelength / height * f
            height, width = float(trgt_sidelength), float(trgt_sidelength)

        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f

        # Build the intrinsic matrices
        full_intrinsic = np.array([[fx, fy], [cx, cy], [width, height]])
        return full_intrinsic

    def _load_pose(self, filename: str) -> torch.Tensor:
        """Load pose from file"""
        lines = open(filename).read().splitlines()
        pose = np.zeros((3, 4), dtype=np.float32)
        for i in range(12):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        pose[:, 2] *= -1
        pose[:, 1] *= -1
        pose = torch.from_numpy(pose.squeeze()).float()
        return pose

    def _load_rgba_image(self, rgba_path: str, bg_color: float) -> torch.Tensor:
        """Load and process RGBA image with background"""
        from PIL import Image
        image = Image.open(rgba_path).convert("RGBA")
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        # Apply background color
        if image.shape[0] == 4:  # RGBA
            alpha = image[3:4]
            rgb = image[:3]
            image = rgb * alpha + bg_color * (1 - alpha)
        else:
            image = image[:3]
        
        return image
