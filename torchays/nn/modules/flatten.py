from typing import Tuple

import torch
from torch import Tensor, nn

from .base import Module


class Flatten(Module, nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)

    def forward_graph(self, input: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None) -> Tuple[Tensor, Tensor]:
        origin_size = self._origin_size(input, weight_graph)
        input_dim = len(origin_size)

        if bias_graph is None:
            bias_graph = torch.zeros_like(input, device=input.device, dtype=input.dtype)
        if weight_graph is None:
            weight_graph = torch.zeros(((*bias_graph.size(), *origin_size)), device=input.device, dtype=input.dtype)

        input_dim = len(self._origin_size(input, weight_graph))
        bias_graph = bias_graph.flatten(self.start_dim, self.end_dim)
        end_dim = self.end_dim
        if end_dim < 0:
            end_dim -= input_dim
        weight_graph = weight_graph.flatten(self.start_dim, end_dim)
        return weight_graph, bias_graph
