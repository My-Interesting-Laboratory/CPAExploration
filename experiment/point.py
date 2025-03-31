import os
from typing import Dict, Iterable, List, Tuple

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
        with default_subplots(
            save_path,
            f"log_{self.name}",
            "count",
            with_grid=False,
            with_legend=False,
        ) as ax:
            ax.bar(nd_x, nd_y, color=color(1), width=0.15, label=f"All Neurons: {sum(nd_y)}")
            ax.bar(id_x, id_y, color=color(0), width=0.15, label=f"Intersect Neurons: {sum(id_y)}")
            ax.legend(prop={"weight": "normal", "size": 7})

    def _ds(self, v: torch.Tensor) -> torch.Tensor:
        return torch.log(v)


class Neurals:
    def __init__(self, *names: str):
        self.neruals: Dict[str, Neural] = dict()
        if names is None:
            return
        for name in names:
            self.neruals[name] = Neural(name)

    def append(self, data_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        for name, data in data_map.items():
            self.neruals[name] = self.neruals.pop(name, Neural(name)).append(*data)
        return self

    def draw_bar(self, dir: str, depth: int):
        for nerual in self.neruals.values():
            nerual.draw_bar(os.path.join(dir, f"{nerual.name}-{depth}.png"))
