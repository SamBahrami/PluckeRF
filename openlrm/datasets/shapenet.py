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


import os
from typing import Union
import random
import numpy as np
import torch
from megfile import smart_path_join

from .base import BaseDataset
from .cam_utils import (
    build_camera_standard,
    build_camera_principle,
    camera_normalization_objaverse,
)
from ..utils.proxy import no_proxy

__all__ = ["ShapenetDataset"]


class ShapenetDataset(BaseDataset):

    def __init__(
        self,
        root_dirs: list[str],
        meta_path: str,
        sample_side_views: int,
        render_image_res_low: int,
        render_image_res_high: int,
        render_region_size: int,
        source_image_res: int,
        normalize_camera: bool,
        normed_dist_to_center: Union[float, str] = None,
        num_all_views: int = 50,
        n_conditioning_views: int = 1,
        same_conditioning_views: bool = False,
    ):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        # go into one of the pose folders and count the number of poses to get the num_all_views
        pose_dir = os.path.join(self.root_dirs[0], self.uids[0], "pose")
        self.num_all_views = len(os.listdir(pose_dir))
        self.n_conditioning_views = n_conditioning_views
        self.same_conditioning_views = same_conditioning_views
        if same_conditioning_views:
            # choose views 64 and 128 as the first 2 sample views
            sample_views = [64, 128, 0, 96, 160, 192, 224, 32, 16, 48, 80, 112, 144, 176, 208]
            # modulo every value in sample views by num_all_views
            sample_views = [view % self.num_all_views for view in sample_views]
            self.sample_views = np.array(sample_views[:self.sample_side_views + 1])

    @staticmethod
    def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
        """Parse intrinsics from intrinsics.txt file for associated object
        in the Shapenet dataset.

        from https://github.com/vsitzmann/scene-representation-networks/blob/8165b500816bb1699f5a34782455f2c4b6d4f35a/geometry.py#L48
        Parameters
        ----------
        filepath : str or Path
            _description_
        trgt_sidelength : _type_, optional
            _description_, by default None
        invert_y : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        # Get camera intrinsics
        with open(filepath, "r") as file:
            f, cx, cy, _ = map(float, file.readline().split())
            grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
            scale = float(file.readline())
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

    @staticmethod
    def _load_pose(filename):
        """from https://github.com/vsitzmann/scene-representation-networks/
        https://github.com/vsitzmann/scene-representation-networks/blob/8165b500816bb1699f5a34782455f2c4b6d4f35a/data_util.py#L43
        """
        lines = open(filename).read().splitlines()
        pose = np.zeros((3, 4), dtype=np.float32)
        for i in range(12):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        pose[:, 2] *= -1
        pose[:, 1] *= -1
        pose = torch.from_numpy(pose.squeeze()).float()
        return pose

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        root_dir = self._locate_datadir(self.root_dirs, uid, locator="intrinsics.txt")

        pose_dir = os.path.join(root_dir, uid, "pose")
        rgba_dir = os.path.join(root_dir, uid, "rgb")
        intrinsics_path = os.path.join(root_dir, uid, "intrinsics.txt")
        im_counter = self.n_conditioning_views

        if self.same_conditioning_views:
            sample_views = self.sample_views
        else:
            # sample views (incl. source view and side views)
            sample_views = np.random.choice(
                range(self.num_all_views), self.sample_side_views + 1, replace=False
            )
        poses, rgbs, bg_colors = [], [], []
        source_image = None
        for view in sample_views:
            pose_path = smart_path_join(pose_dir, f"{view:06d}.txt")
            rgba_path = smart_path_join(rgba_dir, f"{view:06d}.png")
            pose = self._load_pose(pose_path)
            bg_color = random.choice([1.0])
            rgb = self._load_rgba_image(rgba_path, bg_color=bg_color)
            poses.append(pose)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            # Conditioning / source images.
            if source_image is None:
                source_image = self._load_rgba_image(rgba_path, bg_color=1.0)
                im_counter -= 1
                continue
            if im_counter != 0:
                loaded_img = self._load_rgba_image(rgba_path, bg_color=1.0)
                # concatenate onto source image in the channel dimension
                source_image = torch.cat((source_image, loaded_img), dim=1)
                im_counter -= 1
        assert source_image is not None, "Really bad luck!"
        poses = torch.stack(poses, dim=0)
        rgbs = torch.cat(rgbs, dim=0)

        # load intrinsics
        intrinsics = self.parse_intrinsics(intrinsics_path)
        intrinsics = torch.from_numpy(intrinsics).float()

        # Build K intrinsics matrix
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[0, 1], intrinsics[1, 0], intrinsics[1, 1]
        # K_intrinsics_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)

        # build source and target camera features
        source_camera = build_camera_principle(
            poses[:1], intrinsics.unsqueeze(0)
        ).squeeze(0)
        render_camera = build_camera_standard(
            poses, intrinsics.repeat(poses.shape[0], 1, 1)
        )

        K_intrinsics_matrix = render_camera[0, 16:25].view(3, 3)

        # adjust source image resolution
        source_image = torch.nn.functional.interpolate(
            source_image,
            size=(self.source_image_res, self.source_image_res),
            mode="bicubic",
            align_corners=True,
        ).squeeze(0)
        source_image = torch.clamp(source_image, 0, 1)

        # adjust render image resolution and sample intended rendering region
        render_image_res = np.random.randint(
            self.render_image_res_low, self.render_image_res_high + 1
        ) # picks a random resolution between the low and high res
        render_image = torch.nn.functional.interpolate(
            rgbs,
            size=(render_image_res, render_image_res),
            mode="bicubic",
            align_corners=True,
        ) # resizes the image to the random resolution
        render_image = torch.clamp(render_image, 0, 1) # clamps the image to 0 and 1
        anchors = torch.randint(
            0,
            render_image_res - self.render_region_size + 1,
            size=(self.sample_side_views + 1, 2),
        ) # picks a random anchor point for the crop
        crop_indices = torch.arange(
            0, self.render_region_size, device=render_image.device
        ) # creates a tensor of indices for the crop
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(
            -1, self.render_region_size, 1
        )
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(
            -1, 1, self.render_region_size
        ) # creates the indices for the crop
        batch_indices = torch.arange(
            self.sample_side_views + 1, device=render_image.device
        ).view(-1, 1, 1) # creates the batch indices for the crop
        cropped_render_image = render_image[batch_indices, :, index_i, index_j].permute(
            0, 3, 1, 2
        )

        return {
            "uid": uid,
            "source_camera": source_camera,
            "render_camera": render_camera,
            "source_image": source_image,
            "render_image": cropped_render_image,
            "render_anchors": anchors,
            "render_full_resolutions": torch.tensor(
                [[render_image_res]], dtype=torch.float32
            ).repeat(self.sample_side_views + 1, 1),
            "render_bg_colors": torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(
                -1
            ),
            "camera_poses": poses,
            "camera_intrinsics": K_intrinsics_matrix,
        }
