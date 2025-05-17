import multiprocessing as mp
import os
import time
from logging import Logger
from multiprocessing.pool import AsyncResult
from multiprocessing.reduction import ForkingPickler
from typing import Callable, List, Tuple

import numpy as np
import torch

from torchays.nn.modules.base import Tensor

from ..nn import Module
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH
from ..utils import get_logger
from .handler import BaseHandler
from .model import Model
from .optimization import cheby_ball, lineprog_intersect
from .regions import CPACache, CPACacheFactory, CPAFunc, CPAHandler, CPASet, WapperRegion
from .util import check_point, err_callback, find_projection, generate_bound_regions, get_regions, log_time, vertify


class CPAFactory:
    def __init__(
        self,
        workers: int = 1,
        device: torch.device = torch.device("cpu"),
        logger: Logger = None,
        logging: bool = True,
    ):
        self.device = device
        self.logger = logger or get_logger("CPAExploration-Console", multi=(workers > 1))
        self.logging = logging
        self.workers = workers if workers > 1 else 1

    def CPA(
        self,
        net: Model,
        depth: int = -1,
        handler: BaseHandler = None,
        logger: Logger = None,
    ):
        if logger is None:
            logger = self.logger
        return CPA(
            net=net,
            depth=depth,
            handler=handler,
            device=self.device,
            workers=self.workers,
            logger=logger,
            logging=self.logging,
        )

    def __call__(self, *args, **kwds):
        return self.CPA(*args, **kwds)


class CPA:
    """
    CPA needs to ensure that the net has the function:
        >>> def forward_layer(*args, depth=depth):
        >>>     ''' layer is a "int" before every ReLU module. "Layer" can get the layer weight and bias graph.'''
        >>>     if depth == 1:
        >>>         return output
    """

    def __init__(
        self,
        net: Model,
        depth: int = -1,
        handler: BaseHandler = None,
        workers: int = 1,
        device: torch.device = torch.device("cpu"),
        logger: Logger = None,
        logging: bool = True,
    ):
        assert isinstance(net, Module), "the type of net must be \"BaseModule\"."
        self.net = net.graph().to(device)
        self.depth = depth if depth >= 0 else net.depth
        self.cpa_handler = CPAHandler(handler, depth)
        self.device = device
        self.logging = logging
        self.logger = logger
        # muiti-processing
        self.workers, self.pool = 1, None
        if workers > 1:
            self.workers = workers

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"], self_dict["net"], self_dict["cpa_handler"]
        return self_dict

    def start(
        self,
        point: torch.Tensor = None,
        input_size: tuple = (2,),
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
    ) -> int | Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if input_size is None and point is None:
            raise ValueError("input_size and point can not be None at the same time.")
        if point is None:
            return self._start(bounds=bounds, input_size=input_size)
        return self._start_point(point, bounds=bounds)

    def _start_point(
        self,
        point: torch.Tensor,
        *,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        funcs: affine functions which constructs the region.
        region: the funcs > 0?
        neural_funcs: all neural functions in the parent region.
        """
        # Initialize the parameters
        self.net.origin_size = point.size()
        dim = self.net.origin_size.numel()
        # Initialize edges
        p_funcs, p_region, _, self.o_bounds = generate_bound_regions(bounds, dim)
        # Initialize the region set.
        cpa_set, cpa = CPASet(), CPAFunc(p_funcs, p_region, point)
        cpa_set.register(cpa)
        # Start to get point region.
        self.logger.info("Start Get point cpa.")
        start = time.time()
        point_cpa, neural_funcs = self._get_point_cpa(cpa_set)
        self.logger.info(f"End Get point cpa. Times: {time.time()-start}s")
        return point_cpa.funcs, point_cpa.region, neural_funcs

    def _get_point_cpa(self, cpa_set: CPASet) -> Tuple[CPAFunc, torch.Tensor]:
        for p_cpa in cpa_set:
            cpa_cache = self.cpa_handler.cpa_cache_factory().cpa_cache()
            # Get hyperplanes.
            c_funcs = self._functions(p_cpa.point, p_cpa.depth)
            intersect_funcs = self._find_intersect(p_cpa, c_funcs)
            if intersect_funcs is None:
                c_cpa = CPAFunc(p_cpa.funcs, p_cpa.region, p_cpa.point, p_cpa.depth + 1)
                self._nn_region_counts(c_cpa, cpa_set.register, cpa_cache.cpa)
            else:
                c_region = get_regions(p_cpa.point.reshape(1, -1), intersect_funcs)[0]
                # Check and get the child region. Then, the neighbor regions will be found.
                c_cpa, _, _ = self._optimize_child_region(p_cpa, intersect_funcs, c_region)
                if c_cpa is None:
                    return None, None
                # Use the input point to replace the inner point of region.
                c_cpa.point = p_cpa.point
                self._nn_region_counts(c_cpa, cpa_set.register, cpa_cache.cpa)
            # Collect the information of the current parent region including region functions, child functions, intersect functions and number of the child regions.
            cpa_cache.hyperplane(p_cpa, c_funcs, intersect_funcs, 1)
            self.cpa_handler.extend(cpa_cache)
        self.cpa_handler()
        return c_cpa, c_funcs

    def _start(
        self,
        *,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
        input_size: tuple = (2,),
    ) -> int:
        # Initialize the parameters
        self.net.origin_size = torch.Size(input_size)
        dim = self.net.origin_size.numel()
        # Initialize edges
        p_funcs, p_region, p_inner_point, self.o_bounds = generate_bound_regions(bounds, dim)
        # Initialize the region set.
        cpa_set, cpa = CPASet(), CPAFunc(p_funcs, p_region, p_inner_point)
        cpa_set.register(cpa)
        # Start to get the NN regions.
        self.logger.info("Start Get region number.")
        start = time.time()
        counts = self._get_counts(cpa_set)
        self.logger.info(f"CPAFunc counts: {counts}. Times: {time.time()-start}s")
        return counts

    @log_time("CPAFunc counts")
    def _get_counts(self, cpa_set: CPASet) -> int:
        if self.workers == 1:
            return self._single_get_counts(cpa_set)
        # Multi-process
        # Change the ForkingPickler, and stop share memory of torch.Tensor when using multiprocessiong.
        # If share_memory is used, the large number of the fd will be created and lead to OOM.
        _save_reducers = ForkingPickler._extra_reducers
        ForkingPickler._extra_reducers = {}
        self.pool = mp.Pool(processes=self.workers)
        counts = self._multiprocess_get_counts(cpa_set)
        ForkingPickler._extra_reducers = _save_reducers
        return counts

    def _single_get_counts(self, cpa_set: CPASet) -> int:
        counts: int = 0
        for cpa in cpa_set:
            c_funcs = self._functions(cpa.point, cpa.depth)
            self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}, ")
            # Find the child regions or get the region counts.
            count, c_cpa_set, cpa_cache = self._handler_region(cpa, c_funcs, self.cpa_handler.cpa_cache_factory())
            counts += count
            self.cpa_handler.extend(cpa_cache)
            cpa_set.extend(c_cpa_set)
        self.cpa_handler()
        return counts

    def _work(self, cpa: CPAFunc, c_funcs: torch.Tensor, factory: CPACacheFactory):
        self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}, Process-PID: {os.getpid()}. ")
        return self._handler_region(cpa, c_funcs, factory)

    def _multiprocess_get_counts(self, cpa_set: CPASet) -> int:
        """This method of multi-process implementation will result in the inability to use multi-processing when searching the first layer."""
        counts: int = 0

        def callback(args) -> None:
            nonlocal counts
            count, c_cpa_set, cpa_cache = args
            counts += count
            self.cpa_handler.extend(cpa_cache)
            cpa_set.extend(c_cpa_set)

        factory = self.cpa_handler.cpa_cache_factory()
        results: List[AsyncResult] = []
        for cpa in cpa_set:
            # We do not calculate the weigh and bias in the sub-processes.
            # It will use the GPUs and CUDA, and there are many diffcult problems (memory copying, "spawn" model...) that need to be solved.
            c_funcs = self._functions(cpa.point, cpa.depth)

            if cpa.depth == 0:
                # The first layer only has one region.
                # So, we do not need to use multi-processing to search the first layer.
                # We can use the multi-processing to search the child regions.
                self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}.")
                args = self._handler_region(cpa, c_funcs, factory)
                callback(args)
                continue

            # Multi-processing to search the CPAs.
            res = self.pool.apply_async(
                func=self._work,
                args=(cpa, c_funcs, factory),
                callback=callback,
                error_callback=err_callback,
            )
            # Clean finished processes.
            results = [r for r in results if not r.ready()]
            results.append(res)
            if len(cpa_set) != 0:
                continue
            for res in results:
                res.wait()
                if len(cpa_set) > 0:
                    break

        results.clear()
        self.pool.close()
        self.pool.join()
        self.cpa_handler()
        return counts

    def _functions(self, x: torch.Tensor, depth: int):
        # Get the list of the linear functions from DNN.
        functions = self._net_2_cpa(x, depth)
        # Scale the functions.
        return self._scale_functions(functions)

    def _net_2_cpa(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        x = x.float().to(self.device)
        x = x.reshape(*self.net.origin_size).unsqueeze(dim=0)
        with torch.no_grad():
            out: Tensor = self.net.forward_layer(x, depth=depth)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = out.weight_graph, out.bias_graph
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, x.size()[1:].numel())
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1).cpu()

    def _scale_functions(self, functions: torch.Tensor) -> torch.Tensor:
        def _scale_function(function: torch.Tensor):
            v, _ = function.abs().max(dim=0)
            while v < 1:
                function *= 10
                v *= 10
            return function

        for i in range(functions.shape[0]):
            functions[i] = _scale_function(functions[i])
        return functions

    @log_time("Handler region", 2, logging=False)
    def _handler_region(
        self,
        p_cpa: CPAFunc,
        c_funcs: torch.Tensor,
        factory: CPACacheFactory,
    ):
        """
        Search the child regions.
        """
        c_cpa_set, cpa_cache = CPASet(), factory.cpa_cache()
        intersect_funcs, inner_points = self._find_intersect(p_cpa, c_funcs)
        if intersect_funcs is None:
            c_cpa = CPAFunc(p_cpa.funcs, p_cpa.region, p_cpa.point, p_cpa.depth + 1)
            n_regions = 1
            counts = self._nn_region_counts(c_cpa, c_cpa_set.register, cpa_cache.cpa)
        else:
            counts, n_regions = self._search_regions(
                p_cpa=p_cpa,
                intersect_funcs=intersect_funcs,
                inner_points=inner_points,
                c_cpa_set=c_cpa_set,
                cpa_cache=cpa_cache,
            )
        # Collect the information of the current parent region including region functions, child functions, intersect functions and number of the child regions.
        cpa_cache.hyperplane(p_cpa, c_funcs, intersect_funcs, n_regions)
        return counts, c_cpa_set, cpa_cache

    def _search_regions(
        self,
        p_cpa: CPAFunc,
        intersect_funcs: torch.Tensor,
        inner_points: torch.Tensor,
        c_cpa_set: CPASet,
        cpa_cache: CPACache,
    ):

        counts, n_regions = 0, 0
        inner_points: torch.Tensor = torch.cat((inner_points, p_cpa.point.reshape(1, -1)))
        c_regions = get_regions(inner_points, intersect_funcs)
        # Register some regions in WapperRegion for iterate.
        layer_regions = WapperRegion(c_regions)

        def callback(args):
            nonlocal counts, n_regions
            c_cpa, filter_region, neighbor_regions = args
            if c_cpa is None:
                return
            # Add the region to prevent counting again.
            if not layer_regions.update_filter(filter_region):
                return
            # Register new regions for iterate.
            layer_regions.extend(neighbor_regions)
            # Count the number of the regions in the current parent region.
            n_regions += 1
            # Handle the child region.
            counts += self._nn_region_counts(c_cpa, c_cpa_set.register, cpa_cache.cpa)

        # Without multi-processing or the p_cpa is not the first layer.
        # Check and get the child region. Then, the neighbor regions will be found.
        if self.workers == 1 or p_cpa.depth != 0:
            for c_region in layer_regions:
                args = self._optimize_child_region(p_cpa, intersect_funcs, c_region)
                callback(args)
                layer_regions.down()
            return counts, n_regions

        # Multi-processing to search the child regions.
        results: List[AsyncResult] = []
        while len(layer_regions) != 0:
            i = 0
            for c_region in layer_regions:
                res = self.pool.apply_async(
                    func=self._optimize_child_region,
                    args=(p_cpa, intersect_funcs, c_region),
                    callback=callback,
                    error_callback=err_callback,
                )
                results.append(res)
                i += 1
                if i == self.workers:
                    break
            # Wait for all processes to complete
            for res in results:
                res.wait()
            layer_regions.down()
            results.clear()

        return counts, n_regions

    @log_time("Find intersect", 2, logging=False)
    def _find_intersect(self, p_cpa: CPAFunc, funcs: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Find the hyperplanes intersecting the region."""
        intersect_idx, points_idx = [], []
        optim_funcs, optim_x = funcs.numpy(), p_cpa.point.double().numpy()
        pn_funcs = (p_cpa.region.view(-1, 1) * p_cpa.funcs).numpy()
        p_points, w_s = find_projection(p_cpa.point, funcs)
        for i in range(funcs.size(0)):
            if not check_point(p_points[i], w_s[i]):
                continue
            if vertify(p_points[i], p_cpa.funcs, p_cpa.region):
                points_idx.append(i)
                intersect_idx.append(i)
                continue
            if lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds):
                intersect_idx.append(i)
        if len(intersect_idx) == 0:
            return None, None
        intersect_funs, inner_points = funcs[intersect_idx], p_points[points_idx]
        return intersect_funs, inner_points

    def _optimize_child_region(self, p_cpa: CPAFunc, c_funcs: torch.Tensor, c_region: torch.Tensor) -> Tuple[CPAFunc, torch.Tensor, List[torch.Tensor]]:
        """
        1. Check if the region is existed.
        2. Get the neighbor regions and the functions of the region edges.;
        """
        funcs, region = torch.cat([c_funcs, p_cpa.funcs], dim=0), torch.cat([c_region, p_cpa.region], dim=0)
        # ax+b >= 0
        constraint_funcs = region.view(-1, 1) * funcs
        # 1. Check whether the region exists, the inner point will be obtained if existed.
        c_inner_point: torch.Tensor | None = self._find_region_inner_point(constraint_funcs)
        if c_inner_point is None:
            return None, None, None
        # 2. Find the least edges functions to express this region and obtain neighbor regions.
        c_edge_funcs, c_edge_region, filter_region, neighbor_regions = self._optimize_region(funcs, region, constraint_funcs, c_region, c_inner_point)
        c_cpa = CPAFunc(c_edge_funcs, c_edge_region, c_inner_point, p_cpa.depth + 1)
        return c_cpa, filter_region, neighbor_regions

    @log_time("Find inner point", 2, False)
    def _find_region_inner_point(self, functions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the max radius of the insphere in the region to make the radius less than the distance of the insphere center to the all functions
        which can express a liner region.

        minimizing the following function.

        To minimize:
        * min_{x,r} (-r)
        * s.t.  (-Ax+r||A|| <= B)
        *       r > 0
        """
        funcs = functions.numpy()
        x, _, success = cheby_ball(funcs)
        if not success or x is None:
            return None
        return torch.from_numpy(x).float()

    @log_time("Optimize region", 2, False)
    def _optimize_region(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        constraint_funcs: torch.Tensor,
        c_region: torch.Tensor,
        c_inner_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Get the bound hyperplanes which can filter the same regions, and find the neighbor regions."""
        neighbor_regions: List[torch.Tensor] = list()
        filter_region = torch.zeros_like(c_region).type(torch.int8)

        optim_funcs, optim_x = constraint_funcs.numpy(), c_inner_point.double().numpy()
        p_points, _ = find_projection(c_inner_point, funcs)
        c_edge_idx, redundant_idx = [], []
        # TODO: Too much time spent.
        for i in range(optim_funcs.shape[0]):
            # Find the edge hyperplanes by "vertify".
            if not vertify(p_points[i], funcs, region):
                # Use the "lineprog_intersect" to find the edge hyperplanes.
                pn_funcs = np.delete(optim_funcs, [i, *redundant_idx], axis=0)
                success = lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds)
                if not success:
                    redundant_idx.append(i)
                    continue
            c_edge_idx.append(i)
        for i in c_edge_idx:
            if i >= c_region.shape[0]:
                continue
            # Find neighbor regions.
            neighbor_region = c_region.clone()
            neighbor_region[i] = -c_region[i]
            neighbor_regions.append(neighbor_region)
            # Get filter region.
            filter_region[i] = c_region[i]
        c_edge_funcs, c_edge_region = funcs[c_edge_idx], region[c_edge_idx]
        return c_edge_funcs, c_edge_region, filter_region, neighbor_regions

    def _nn_region_counts(self, cpa: CPAFunc, set_register: Callable[[CPAFunc], None], cpa_register: Callable[[CPAFunc], bool]) -> int:
        # If current layer is the last layer, the region is the in count.
        # Collect the information of the final region.
        if cpa_register(cpa):
            return 1
        # If not the last layer, the region will be parent region in the next layer.
        set_register(cpa)
        return 0
