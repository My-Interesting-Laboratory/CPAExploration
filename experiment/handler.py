from typing import Dict, List

import torch

from torchays import nn
from torchays.cpa import BaseHandler

from .hpa import HyperplaneArrangement


class Handler(BaseHandler):
    def __init__(self, with_regions: bool, with_hpas: bool) -> None:
        self.with_region = with_regions
        self.with_hpas = with_hpas
        if with_regions:
            self._init_region()
        if with_hpas:
            self._init_inner_hyperplanes

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


class TrainHandler:
    def step_handler(self, net: nn.Module, epoch: int, step: int, total_step: int, loss: torch.Tensor, acc: float):
        raise NotImplementedError()

    def epoch_handler(self, net: nn.Module, epoch: int, loss: torch.Tensor, acc: float):
        raise NotImplementedError()
