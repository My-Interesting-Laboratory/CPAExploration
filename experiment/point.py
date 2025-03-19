from typing import List

import torch

from torchays.graph import color, default_subplots

from .draw import bar


class Neural:
    neural: List[torch.Tensor]
    inter: List[torch.Tensor]

    def __init__(self, name: str):
        self.name = name
        self.inter = list()
        self.neural = list()

    def append(self, neural: torch.Tensor = None, inter: torch.Tensor = None):
        if neural is not None:
            self.neural.append(neural)
        if inter is not None:
            self.inter.append(inter)
        return self

    def draw_bar(self, save_path: str):
        nd_x, nd_y = bar(self.neural, 0.2, self._ds)
        id_x, id_y = bar(self.inter, 0.2, self._ds)
        with default_subplots(save_path, f"log_{self.name}", "count", with_grid=False, with_legend=False) as ax:
            ax.bar(nd_x, nd_y, color=color(1), width=0.15, label=f"All Neurons: {sum(nd_y)}")
            ax.bar(id_x, id_y, color=color(0), width=0.15, label=f"Intersect Neurons: {sum(id_y)}")
            ax.legend(prop={"weight": "normal", "size": 7})

    def _ds(self, v: torch.Tensor) -> torch.Tensor:
        return torch.log(v)


class Neurals:
    def __init__(self):
        self.ds = Neural("distance")
        self.v = Neural("values")
        self.w_s = Neural("weights")

    def append(
        self,
        nerual_ds: torch.Tensor,
        inter_ds: torch.Tensor,
        nerual_v: torch.Tensor,
        inter_v: torch.Tensor,
        nerual_ws: torch.Tensor,
        inter_ws: torch.Tensor,
    ):
        self.ds.append(nerual_ds, inter_ds)
        self.v.append(nerual_v, inter_v)
        self.w_s.append(nerual_ws, inter_ws)
        return self
