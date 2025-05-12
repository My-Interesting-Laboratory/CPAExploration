import torch
from torch import Tensor, nn

from .base import Module, check_graph, set_graph


class Flatten(Module, nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)

    def forward_graph(self, input: Tensor) -> Tensor:

        output = nn.Flatten.forward(self, input)
        input, weight_graph, bias_graph = check_graph(input)

        origin_size = self._origin_size(input)
        input_dim = len(origin_size)

        if bias_graph is None:
            bias_graph = torch.zeros_like(input, device=input.device, dtype=input.dtype)
        if weight_graph is None:
            weight_graph = torch.zeros(((*bias_graph.size(), *origin_size)), device=input.device, dtype=input.dtype)

        input_dim = len(origin_size)
        bias_graph = bias_graph.flatten(self.start_dim, self.end_dim)
        end_dim = self.end_dim
        if end_dim < 0:
            end_dim -= input_dim
        weight_graph = weight_graph.flatten(self.start_dim, end_dim)
        return set_graph(output, weight_graph, bias_graph)
