from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import Module, check_graph, set_graph, get_size_to_one


class ParamReLU(Module):
    def __init__(self, active_slope: float = 1.0, negative_slope: float = 0.0) -> None:
        super().__init__()
        self._active_slope = active_slope
        self._negative_slope = negative_slope

    def _forward_graph(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight_graph, bias_graph = check_graph(input)

        origin_size = self._origin_size(input)
        graph_size = input.weight_graph.size()
        # ((*input.shape)), (*origin_size)), ((*input.shape))
        wg, bg = torch.zeros(graph_size, device=input.device, dtype=input.dtype), torch.zeros(*input.size(), device=input.device, dtype=input.dtype)
        active_slope = torch.ones((1), device=input.device, dtype=input.dtype) * self._active_slope
        negative_slope = torch.ones((1), device=input.device, dtype=input.dtype) * self._negative_slope
        x_relu_hot = torch.where(input > 0, active_slope, negative_slope)
        wg += x_relu_hot.view(*x_relu_hot.size(), *get_size_to_one(origin_size))
        bg += x_relu_hot
        if weight_graph is None:
            weight_graph, bias_graph = 1, 0
        wg *= weight_graph
        bg *= bias_graph
        return wg, bg


class ReLU(ParamReLU, nn.ReLU):
    __doc__ = nn.ReLU.__doc__

    def __init__(self, inplace: bool = False) -> None:
        nn.ReLU.__init__(self, inplace)
        ParamReLU.__init__(self, 1, 0)

    def forward_graph(self, input: Tensor):
        output = nn.ReLU.forward(self, input)
        weight_graph, bias_graph = ParamReLU._forward_graph(self, input)
        return set_graph(output, weight_graph, bias_graph)


class LeakyRule(ParamReLU, nn.LeakyReLU):
    __doc__ = nn.LeakyReLU.__doc__

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        nn.LeakyReLU.__init__(self, negative_slope, inplace)
        ParamReLU.__init__(self, 1, negative_slope)

    def forward_graph(self, input):
        output = nn.LeakyReLU.forward(self, input)
        weight_graph, bias_graph = ParamReLU._forward_graph(self, input)
        return set_graph(output, weight_graph, bias_graph)
