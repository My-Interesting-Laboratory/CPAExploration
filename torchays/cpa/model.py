from typing import Any, Dict, Tuple

import torch
from torch.nn import Parameter

from torchays import graph

from ..nn import Linear, Module
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH


class Model(Module):
    n_relu: int
    name: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = __class__.__name__

    def forward_layer(self, x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        raise NotImplementedError()


class ProjectWrapper(Model):
    def __init__(
        self,
        model: Model,
        proj_dims: tuple | None = None,
        proj_values: torch.Tensor = None,
    ):
        super().__init__()
        self.model = model
        self.n_relu = model.n_relu
        self.name = model.name
        self._wrapper = (proj_dims and proj_values) is not None
        if self._wrapper:
            self._input_size = proj_values.size()
            self.input_size = (len(proj_dims),)
            self.proj_dim = proj_dims
            self.set_project(proj_dims, proj_values)

    def set_project(self, proj_dims: tuple | None, proj_values: torch.Tensor):
        num_proj_dims = len(proj_dims)
        neruals = proj_values.size().numel()
        self.wrapper_layers = Linear(num_proj_dims, neruals)
        _weight = torch.zeros_like(self.wrapper_layers.weight)
        _bias = proj_values.view(-1).detach().float()
        for i in range(num_proj_dims):
            dim = proj_dims[i]
            # bias is 0, weight is 1 for project dim.
            _weight[dim][i] = 1
            # others, weight is 0，bias is input；
            _bias[dim] = 0
        self.wrapper_layers.weight = Parameter(_weight)
        self.wrapper_layers.bias = Parameter(_bias)

    def _forward(self, x):
        if self._wrapper and self.graphing:
            # when finding CPAs
            x, graph = self.wrapper_layers(x)
            x: torch.Tensor = x.view(x.size(0), *self._input_size)
            wg, bg = graph[WEIGHT_GRAPH], graph[BIAS_GRAPH]
            wg: torch.Tensor = wg.view(x.size(0), *self._input_size, *self.input_size)
            bg: torch.Tensor = bg.view(x.size(0), *self._input_size)
            x = (x, {WEIGHT_GRAPH: wg, BIAS_GRAPH: bg})
        return x

    def forward(self, x):
        return self.model(self._forward(x))

    def forward_layer(self, x, depth):
        return self.model.forward_layer(self._forward(x), depth)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict, assign)
