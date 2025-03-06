from typing import Dict, List

import torch

from torchays.cpa import BaseHandler

from .hpa import HyperplaneArrangement


class Handler(BaseHandler):
    def __init__(self) -> None:
        self._init_region()._init_inner_hyperplanes()

    def _init_region(self):
        self.funs = list()
        self.regions = list()
        self.points = list()
        return self

    def region(
        self,
        fun: torch.Tensor,
        region: torch.Tensor,
        point: torch.Tensor,
    ) -> None:
        self.funs.append(fun.cpu().numpy())
        self.regions.append(region.cpu().numpy())
        self.points.append(point.cpu().numpy())

    def _init_inner_hyperplanes(self):
        # {depth: [HyperplaneArrangement,]}
        self.hyperplane_arrangements: Dict[int, List[HyperplaneArrangement]] = dict()
        return self

    def inner_hyperplanes(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        depth_hp_arrs = self.hyperplane_arrangements.pop(depth, list())
        hp_arr = HyperplaneArrangement(p_funs, p_regions, c_funs, intersect_funs, n_regions, depth)
        depth_hp_arrs.append(hp_arr)
        self.hyperplane_arrangements[depth] = depth_hp_arrs
