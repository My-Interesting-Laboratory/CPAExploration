from collections import deque
from math import floor
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import threading
from typing import Deque, List, Tuple

import torch

from .handler import BaseHandler


class CPAFunc:
    def __init__(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        point: torch.Tensor,
        depth: int = 0,
    ):
        self.funcs = funcs
        self.region = region
        self.point = point
        self.depth = depth


class CPASet:
    def __init__(self) -> None:
        self._cpas: Deque[CPAFunc] = deque()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._cpas.popleft()
        except:
            raise StopIteration("CPASet next error")

    def __len__(self) -> int:
        return len(self._cpas)

    def register(
        self,
        region: CPAFunc,
    ):
        self._cpas.append(region)

    def extend(self, cpa_set):
        self._cpas.extend(cpa_set._cpas)

    def __str__(self):
        return self._cpas.__str__()


class WapperRegion:
    """
    Get the region of the function list.
    *  1 : f(x) > 0
    * -1 : f(x) <= 0
    """

    # The maximum capacity of a tensor.
    # If this value is reached, the calculation speed will slow down.
    _upper = 32767

    def __init__(self, regions: torch.Tensor):
        self.filters: Deque[torch.Tensor] = deque()
        self.filters.appendleft(torch.Tensor().type(torch.int8))
        self.regions: Deque[torch.Tensor] = deque()
        self._up_size = floor(self._upper / regions.size(1))
        self.cache: Deque[torch.Tensor] = deque()
        self.extend(regions).down()

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        try:
            region = self.regions.popleft()
            while self._check(region):
                region = self.regions.popleft()
            return region
        except:
            raise StopIteration("WapperRegion next error")

    def _update_list(self):
        self.output: torch.Tensor = self.regions.popleft()

    def _check(self, region: torch.Tensor):
        """Check if the region has been used."""
        try:
            including = False
            for filter in self.filters:
                res = ((filter.abs() * region) - filter).abs().sum(dim=1)
                if 0 in res:
                    return True
            return including
        except:
            return False

    def update_filter(self, region: torch.Tensor) -> bool:
        if self._check(region):
            return False
        if self.filters[0].size(0) == self._up_size:
            self.filters.appendleft(torch.Tensor().type(torch.int8))
        self.filters[0] = torch.cat([self.filters[0], region.unsqueeze(0)], dim=0)
        return True

    def register(self, region: torch.Tensor):
        if not self._check(region):
            self.regions.append(region)

    def extend(self, regions: List[torch.Tensor]):
        self.cache.extend(regions)
        return self

    def down(self):
        while len(self.cache) != 0:
            self.register(self.cache.pop())


class CPACacheFactory:
    def __init__(self, with_regions: bool, with_hpas: bool, depth: int):
        self.depth = depth
        self.with_regions = with_regions
        self.with_hpas = with_hpas

    def cpa_cache(self):
        return CPACache(self.with_regions, self.with_hpas, self.depth)


class CPACache:
    def __init__(self, with_regions: bool, with_hpas: bool, depth: int):
        self.depth = depth
        self.cpa, self.hyperplane = self._empty_cpa, self._empty
        if with_regions:
            self.cpas: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque()
            self.cpa = self._cpa
        if with_hpas:
            self.hyperplanes: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = deque()
            self.hyperplane = self._hyperplane

    def _empty(self, *args, **kargs):
        return

    def _empty_cpa(self, cpa: CPAFunc):
        return cpa.depth > self.depth

    def _cpa(self, cpa: CPAFunc) -> bool:
        if cpa.depth > self.depth:
            self.cpas.append((cpa.funcs, cpa.region, cpa.point))
            return True
        return False

    def _hyperplane(
        self,
        p_cpa: CPAFunc,
        c_funcs: torch.Tensor,
        intersection_funcs: torch.Tensor,
        n_regions: int,
    ):
        self.hyperplanes.append((p_cpa.funcs, p_cpa.region, c_funcs, intersection_funcs, n_regions, p_cpa.depth))


class CPAHandler(object):
    def __init__(self, handler: BaseHandler, depth: int):
        self.handler = handler if handler is not None else BaseHandler()
        self.depth = depth
        self.cpas: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque()
        self.hyperplanes: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = deque()

    def __call__(self):
        if self.handler.with_region:
            for funcs, region, inner_point in self.cpas:
                self.handler.region(funcs, region, inner_point)
            self.cpas.clear()
        if self.handler.with_hpas:
            for args in self.hyperplanes:
                self.handler.inner_hyperplanes(*args)
            self.hyperplanes.clear()

    def extend(self, cache: CPACache):
        if self.handler.with_region:
            self.cpas.extend(cache.cpas)
        if self.handler.with_hpas:
            self.hyperplanes.extend(cache.hyperplanes)

    def cpa_cache_factory(self):
        return CPACacheFactory(
            self.handler.with_region,
            self.handler.with_hpas,
            self.depth,
        )
