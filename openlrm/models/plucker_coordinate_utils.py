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
from torch import linalg

def create_orthogonal_triplane_rays(num_rays=16) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create rays that lie on the xy, xz, yz planes of the unit cube,
    i.e. the SceneBox for the nerf scene. The rays are evenly spaced
    across the planes, in the 'centre' of the pixels.

    Parameters
    ----------
    num_rays : int, optional
        the number of rays to generate on each plane, by default 16

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        origins and directions of the rays
    """
    # And in another way
    uv = torch.stack(torch.meshgrid(
        torch.arange(num_rays, dtype=torch.float32),
        torch.arange(num_rays, dtype=torch.float32),
        indexing='ij',
    ))
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    x_cam = (uv[:, 0] * (1./num_rays) + (0.5/num_rays))
    y_cam = (uv[:, 1] * (1./num_rays) + (0.5/num_rays))
    z_cam = torch.zeros_like(x_cam)
    # convert this to an array of origins
    xy_origins = torch.stack((x_cam, y_cam , z_cam), dim=-1)
    xz_origins = torch.stack((x_cam, z_cam, y_cam), dim=-1)
    yz_origins = torch.stack((z_cam, y_cam, x_cam), dim=-1)
    xy_direction = torch.tensor([0.0, 0.0, 1.0])
    xy_directions = torch.stack([xy_direction] * xy_origins.shape[0])
    xz_direction = torch.tensor([0.0, -1.0, 0.0])
    xz_directions = torch.stack([xz_direction] * xz_origins.shape[0])
    yz_direction = torch.tensor([-1.0, 0.0, 0.0])
    yz_directions = torch.stack([yz_direction] * yz_origins.shape[0])

    alt_origins_list = [xy_origins, xz_origins, yz_origins]
    alt_directions_list = [xy_directions, xz_directions, yz_directions]
    # scale origins to be within -1 and 1

    alt_origins_list[0] = alt_origins_list[0] * 2 - 1
    alt_origins_list[1] = alt_origins_list[1] * 2 - 1
    alt_origins_list[2] = alt_origins_list[2] * 2 - 1
    return alt_origins_list, alt_directions_list


def to_plucker_coordinates(direction, origin):
    """Calculate the Plucker coordinates for a ray given its direction and origin."""
    # Calculate the Plucker coordinates
    pluckers = []
    for i in range(len(direction)):
        moment = linalg.cross(origin[i], direction[i])
        plucker = torch.cat((direction[i], moment), dim=-1)
        pluckers.append(plucker)
    return pluckers


def plucker_coordinates_for_triplanes(sample_size: int = 32, square_representation: bool = True) -> torch.Tensor:
    triplane_origins, triplane_directions = create_orthogonal_triplane_rays(num_rays=sample_size)
    xy, xz, yz = to_plucker_coordinates(triplane_directions, triplane_origins)
    if square_representation:
        # For a square repesentation of the triplanes, i.e. shape [sample_size * sample_size, 18]
        plucker_coordinates = torch.cat([xy, xz, yz], dim=1)
    else:
        # For a rectangular representation [sample_size, sample_size * 3, 6]
        plucker_coordinates = torch.cat([xy, xz, yz], dim=0)
    return plucker_coordinates.unsqueeze(0)


def plucker_distance_calculation(q_pluckers: torch.Tensor, k_pluckers: torch.Tensor) -> torch.Tensor:
    """Calculate the distance between all these plucker coordinates
    for the cross attention layer
    Plucker coordinates are assumed to be in the format [d, m] where d is the direction
    and m is the moment of the ray. d and m are both in R^3.
    """
    q_d = q_pluckers[..., :3]
    q_m = q_pluckers[..., 3:]
    k_d = k_pluckers[..., :3]
    k_m = k_pluckers[..., 3:]
    # Calculate dot product, output shape of [batch, q_d.shape[1], k_m.shape[1]]
    # dot_q_d_k_m = torch.einsum("bik,bjk->bij", q_d, k_m)
    # dot_k_d_q_m = torch.einsum("bik,bjk->bij", q_m, k_d)
    
    numerator = torch.abs(torch.einsum("bik,bjk->bij", q_d, k_m) + torch.einsum("bik,bjk->bij", q_m, k_d))

    # # Lets try it the naiive way without einsum and see if we get the same results
    # naiive_numerator = torch.zeros_like(numerator)
    # for b in range(q_d.shape[0]):
    #     for i in range(q_d.shape[1]):
    #         for j in range(k_d.shape[1]):
    #             naiive_numerator[b, i, j] = torch.abs(torch.dot(q_d[b, i], k_m[b, j]) + torch.dot(q_m[b, i], k_d[b, j]))


    # if torch.allclose(naiive_numerator, numerator):
    #     pass

    denominator = torch.norm(torch.linalg.cross(q_d.unsqueeze(2), k_d.unsqueeze(1)), dim=-1)
    ray_distance = numerator / denominator

    # Find values where denonimator is 0, apply other ray_distance calculation
    same_d = torch.where(denominator == 0, 1.0, 0.0)
    # count the number of values where the denominator is 0
    same_d_count = same_d.sum()
    same_d_vector_indexes = torch.nonzero(same_d)
    # Assuming same_d_vector_indexes is a tensor of shape (N, 3)
    # get the batch, index in ca and index in triplanes variables which are the same
    b, q_i, triplanes_j = same_d_vector_indexes.unbind(dim=-1)
    # Calculate the alternative numerators and denominators
    d_vector = q_d[b, q_i]
    m_vector = q_m[b, q_i]
    other_m = k_m[b, triplanes_j]
    other_d = k_d[b, triplanes_j]
    alternative_numerator = torch.norm(torch.linalg.cross(d_vector, (other_m - m_vector)), dim=-1)
    # In the math this is squared, but we have unit vectors so it doesnt matter.
    alternative_denominator = torch.norm(other_d, dim=-1) 
    denominator_with_other_d = torch.norm(d_vector, dim=-1)
    # Take the minimum value from the two different denominators. If 0, then we
    # are comparing a ray with one of the base triplane 0 rays, 
    # which we want to force the distance to be 0
    denominator = torch.min(alternative_denominator, denominator_with_other_d)
    ray_distance[b, q_i, triplanes_j] = alternative_numerator / alternative_denominator

    # Set inf values to 0, these only occur for class token?
    ray_distance = torch.nan_to_num(ray_distance, nan=0.0, posinf=0.0)
    return torch.transpose(ray_distance, 1, 2)