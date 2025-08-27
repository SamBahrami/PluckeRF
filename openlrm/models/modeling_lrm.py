# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from accelerate.logging import get_logger

from .embedder import CameraEmbedder
from .transformer import TransformerDecoder
from .rendering.synthesizer import TriplaneSynthesizer
from .plucker_coordinate_utils import plucker_coordinates_for_triplanes
from .rendering.utils.ray_sampler import RaySampler
import math

logger = get_logger(__name__)


class ModelLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, camera_embed_dim: int, rendering_samples_per_ray: int,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768,
                 model_type: str = "camera_modulation"):
        super().__init__()
        """Model type can be
            - "camera_modulation": camera modulation (OpenLRM 1 view)
            - "explicit": biased attention with explicit distance calculation for attention mechanism
            - "implicit": attention mechanism without bias
        """
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.model_type = model_type


        # Base triplanes and a projection for the input to the transformer
        self.base_plucker_triplanes = None 
        self.base_triplanes_projection = nn.Linear(6, transformer_dim)
        self.base_triplanes_projection.bias.data.fill_(0)
        self.base_triplanes_projection.weight.data[:6, :6].copy_(torch.eye(6))
        # # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        # self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)

        # Set model type and their specific modules as required
        if self.model_type == "no_camera_modulation":
            self.transformer = TransformerDecoder(
                block_type='cond',
                num_layers=transformer_layers, num_heads=transformer_heads,
                inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=None,
            )
        elif self.model_type == "camera_modulation":
            self.camera_embedder = CameraEmbedder(
                raw_dim=12+4, embed_dim=camera_embed_dim,
            )
            self.transformer = TransformerDecoder(
                block_type='cond_mod',
                num_layers=transformer_layers, num_heads=transformer_heads,
                inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=camera_embed_dim,
            )
        elif self.model_type == "implicit":
            self.transformer = TransformerDecoder(
                block_type='cond',
                num_layers=transformer_layers, num_heads=transformer_heads,
                inner_dim=transformer_dim, cond_dim=encoder_feat_dim+6, mod_dim=None,
            )
            self.ray_sampler = RaySampler()
        elif self.model_type == "explicit":
            self.transformer = TransformerDecoder(
                block_type='cond_explicit',
                num_layers=transformer_layers, num_heads=transformer_heads,
                inner_dim=transformer_dim, cond_dim=encoder_feat_dim+6, mod_dim=None,
                triplane_dim=triplane_low_res,
            )
            self.ray_sampler = RaySampler()
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=2, stride=2, padding=0)
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, samples_per_ray=rendering_samples_per_ray, model_type=model_type
        )

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ['dino', 'dinov2'], "Unsupported encoder type"
        if encoder_type == 'dino':
            from .encoders.dino_wrapper import DinoWrapper
            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == 'dinov2':
            from .encoders.dinov2_wrapper import Dinov2Wrapper
            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper

    def forward_transformer(self, image_feats, camera_embeddings):
        N = image_feats.shape[0]
        
        if self.base_plucker_triplanes is None:
            self.base_plucker_triplanes = plucker_coordinates_for_triplanes(self.triplane_low_res, False).to(image_feats.device)
        x = self.base_triplanes_projection(self.base_plucker_triplanes)
        
        x = x.repeat(N, 1, 1) # [N, L, D]
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    @torch.compile
    def forward_planes(self, image, camera, camera_K_intrinsics=None, camera_poses = None, render_cameras=None):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        # camera_K_intrinsics: [N, 3, 3]
        # camera_poses: [N, k_images, 3, 4]
        N = image.shape[0]

        # encode image
        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
        # transformer generating planes
        if self.model_type == "no_camera_modulation":
            tokens = self.forward_transformer(image_feats, None)
        elif self.model_type == "camera_modulation":
            # embed camera
            camera_embeddings = self.camera_embedder(camera)
            assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
                f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"
            tokens = self.forward_transformer(image_feats, camera_embeddings)
        elif self.model_type == "implicit" or self.model_type == "explicit":
            # Get the first k images that you use as input images
            # Camera poses only takes the number of images used as input images.
            render_cameras = render_cameras[:, :camera_poses.shape[1], :]
            N, M = render_cameras.shape[:2]
            cam2world_matrix = render_cameras[..., :16].view(N, M, 4, 4)
            intrinsics = render_cameras[..., 16:25].view(N, M, 3, 3)

            # Anchor to the top left corner of the image
            anchor = torch.tensor([[0., 0.]])
            anchors = anchor.repeat(N*M, 1).to(image_feats.device)
            # Region size and resolution are the same. We want to get a ray for every single
            # pixel from the very corner of our image, corresponding to the shape of our
            # image features.
            region_size = int(math.sqrt(image_feats.shape[1]// render_cameras.shape[1]))
            resolutions = torch.tensor([[region_size]]).repeat(N*M, 1).to(image_feats.device)
            ray_origins, ray_directions = self.ray_sampler(
                cam2world_matrix=cam2world_matrix.reshape(-1, 4, 4),
                intrinsics=intrinsics.reshape(-1, 3, 3),
                resolutions=resolutions,
                anchors=anchors,
                region_size=region_size,
            )

            ray_origins, ray_directions = ray_origins.view(N, M, -1, 3), ray_directions.view(N, M, -1, 3)
            plucker_m = torch.cross(ray_origins, ray_directions, dim=-1)
            # concatenate with ray_directions
            plucker_coordinates = torch.cat([ray_directions, plucker_m], dim=-1)
            # Prepend Dino each with a plucker coordinate of all zeros (CLS Token)
            if self.encoder.__class__.__name__.startswith("Dino"):
                plucker_coordinates = torch.cat([torch.zeros(N, M, 1, 6).to(plucker_coordinates.device), plucker_coordinates], dim=2)
            plucker_coordinates = plucker_coordinates.view(N, M*(plucker_coordinates.shape[2]), 6)

            image_feats = torch.cat([image_feats, plucker_coordinates], dim=-1)
            tokens = self.forward_transformer(image_feats, None)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"
        return planes, None

    def forward(self, image, source_camera, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size: int, camera_K_intrinsics = None, camera_poses = None):
        # image: [N, C_img, H_img, W_img]
        # source_camera: [N, D_cam_raw]
        # render_cameras: [N, M, D_cam_render]
        # render_anchors: [N, M, 2]
        # render_resolutions: [N, M, 1]
        # render_bg_colors: [N, M, 1]
        # render_region_size: int
        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        assert image.shape[0] == render_anchors.shape[0], "Batch size mismatch for image and render_anchors"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        N, M = render_cameras.shape[:2]
        planes, image_feats = self.forward_planes(image, source_camera, camera_K_intrinsics, camera_poses, render_cameras)

        # Gamma values to log in the transformer
        other_metrics = {}
        if self.model_type == "explicit":
            for i, layer in enumerate(self.transformer.layers):
                other_metrics[f"gamma_ca_{i}"] = layer.gamma_ca.item()
                other_metrics[f"gamma_sa_{i}"] = layer.gamma_sa.item()

        # render target views
        render_results = self.synthesizer(planes, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size, dinov2_image_features=image_feats)
        assert render_results['images_rgb'].shape[0] == N, "Batch size mismatch for render_results"
        assert render_results['images_rgb'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

        return {
            'planes': planes,
            **render_results,
            **other_metrics
        }
