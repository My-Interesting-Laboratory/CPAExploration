import torch
import torch.nn as nn
from torch import Tensor

from .base import Module, check_graph, set_graph


class Linear(Module, nn.Linear):
    __doc__ = nn.Linear.__doc__

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward_graph(self, input: Tensor) -> Tensor:
        """
        Analyzing linear \n
        graph_size: (n, out_feature, (*origin_size)))
        """
        output = nn.Linear.forward(self, input)
        input, weight_graph, bias_graph = check_graph(input)
        # bias_graph
        bias_graph = torch.zeros_like(input, device=input.device, dtype=input.dtype) if bias_graph is None else bias_graph
        bias_graph = nn.Linear.forward(self, bias_graph)
        # weight_graph
        origin_size = self._origin_size(input)
        if weight_graph is None:
            weight_graph = torch.zeros(((*bias_graph.size(), *origin_size)), device=input.device, dtype=input.dtype)
            weight_graph += self.weight
        else:
            weight_graph = weight_graph.reshape(-1, self.in_features, origin_size.numel())
            weight_graph = torch.einsum("nis,oi -> nos", [weight_graph, self.weight]).reshape(-1, self.out_features, *origin_size)
        return set_graph(output, weight_graph, bias_graph)
