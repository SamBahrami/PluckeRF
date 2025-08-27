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


from functools import partial
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from .plucker_coordinate_utils import plucker_distance_calculation, plucker_coordinates_for_triplanes


logger = get_logger(__name__)


class TransformerDecoder(nn.Module):

    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    def __init__(self, block_type: str,
                 num_layers: int, num_heads: int,
                 inner_dim: int, cond_dim: int = None, mod_dim: int = None,
                 eps: float = 1e-6, triplane_dim: int = 32):
        super().__init__()
        self.block_type = block_type
        self.triplane_dim = triplane_dim
        self.base_triplanes = None
        self.sa_distance_matrix = None
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim, mod_dim)(
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in ['basic', 'cond', 'mod', 'cond_mod', 'cond_explicit'], \
            f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == 'basic':
            assert cond_dim is None and mod_dim is None, \
                f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock
            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == 'cond':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionBlock"
            assert mod_dim is None, f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock
            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == 'mod':
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(f"modulation without condition is not implemented")
        elif self.block_type == 'cond_mod':
            assert cond_dim is not None and mod_dim is not None, \
                f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock
            logger.debug(f"Using ConditionModulationBlock")
            return partial(ConditionModulationBlock, inner_dim=inner_dim, cond_dim=cond_dim, mod_dim=mod_dim)
        elif self.block_type == 'cond_explicit':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionExplicitattnBlock"
            assert mod_dim is None, f"Modulation dimension is not supported for ConditionExplicitattnBlock"
            from .block import ConditionExplicitattnBlock
            logger.debug(f"Using ConditionExplicitattnBlock")
            return partial(ConditionExplicitattnBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        else:
            raise ValueError(f"Unsupported block type during runtime: {self.block_type}")

    def assert_runtime_integrity(self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == 'basic':
            assert cond is None and mod is None, \
                f"Condition and modulation are not supported for BasicBlock"
        elif self.block_type == 'cond':
            assert cond is not None and mod is None, \
                f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == 'cond_explicit':
            assert cond is not None and mod is None, \
                f"Condition must be specified and modulation is not supported for ConditionExplicitattnBlock"
        elif self.block_type == 'mod':
            raise NotImplementedError(f"modulation without condition is not implemented")
        else:
            assert cond is not None and mod is not None, \
                f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor, sa_distance_matrix:torch.Tensor, ca_distance_matrix:torch.Tensor):
        if self.block_type == 'basic':
            return layer(x)
        elif self.block_type == 'cond_explicit':
            return layer(x, cond, sa_distance_matrix, ca_distance_matrix)
        elif self.block_type == 'cond':
            return layer(x, cond)
        elif self.block_type == 'mod':
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        # mod: [N, D_mod] or None
        self.assert_runtime_integrity(x, cond, mod)

        if self.block_type == 'cond_explicit':
            # No gradients, just distance.
            with torch.no_grad():
                if self.base_triplanes is None:
                    self.base_triplanes = plucker_coordinates_for_triplanes(self.triplane_dim, False).to(x.device)
                    self.sa_distance_matrix = plucker_distance_calculation(self.base_triplanes, self.base_triplanes)
                    # Repeat the distance matrix and base_triplanes for the batch size
                    # self.base_triplanes = self.base_triplanes.repeat(x.shape[0], 1, 1).to(x.device)
                    # self.sa_distance_matrix = self.sa_distance_matrix.repeat(x.shape[0], 1, 1).to(x.device)

                base_triplanes = self.base_triplanes.repeat(x.shape[0], 1, 1).to(x.device)
                # Calculate closeness matrices
                ca_distance_matrix = plucker_distance_calculation(cond[..., -6:], base_triplanes)
                sa_distance_matrix = self.sa_distance_matrix.repeat(x.shape[0], 1, 1).to(x.device)

            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod, sa_distance_matrix, ca_distance_matrix)
        else:
            sa_distance_matrix = None
            ca_distance_matrix = None

            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod, sa_distance_matrix, ca_distance_matrix)
        x = self.norm(x)
        return x
