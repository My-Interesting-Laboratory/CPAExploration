import math
import os
from typing import Callable, Dict, List, Tuple

import matplotlib
import matplotlib.axis
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch

from torchays import nn
from torchays.graph import COLOR, color, plot_regions, plot_regions_3d, default_subplots


def bar(
    values: List[torch.Tensor],
    interval: float = 0.2,
    callable: Callable[[torch.Tensor], torch.Tensor] = None,
) -> Tuple[List[float], List[int]]:
    values = torch.cat(values).reshape(-1)
    counts: Dict[float, int] = dict()
    # interval
    for v in values:
        if callable is not None:
            v = callable(v)
        k = math.floor((v / interval + 0.5)) * interval
        count = counts.pop(k, 0) + 1
        counts[k] = count
    x = sorted(counts)
    y = [counts.get(k) for k in x]
    return x, y


class DrawRegionImage:
    def __init__(
        self,
        region_num: int,
        funcs: np.ndarray,
        regions: np.ndarray,
        points: np.ndarray,
        save_dir: str,
        net: nn.Module,
        n_classes=2,
        bounds=(-1, 1),
        device=torch.device("cpu"),
        with_ticks=True,
    ) -> None:
        self.region_num = region_num
        self.funcs = funcs
        self.regions = regions
        self.points = points
        self.save_dir = save_dir
        self.net = net.to(device).eval()
        self.n_classes = n_classes
        self.bounds = bounds
        self.min_bound, self.max_bound = bounds
        self.device = device
        self.with_ticks = with_ticks

    def draw(self, img_3d: bool = False):
        draw_funs = [self.draw_region_img, self.draw_region_img_result]
        if img_3d:
            draw_funs.append(self.draw_region_img_3d)
        for draw_fun in draw_funs:
            try:
                draw_fun()
            except Exception as e:
                print(f"Warning: {draw_fun.__name__} is not supported. Error: {e}")

    def draw_region_img(self, file_name="region_img.png"):
        with self._ax(file_name) as ax:
            plot_regions(
                self.funcs,
                self.regions,
                ax=ax,
                color=color,
                edgecolor="gray",
                linewidth=0.1,
                xlim=self.bounds,
                ylim=self.bounds,
            )
            self._set_axis(ax)

    def _z_fun(self, xy: np.ndarray) -> Tuple[np.ndarray, int]:
        xy = torch.from_numpy(xy).to(self.device).float()
        z: torch.Tensor = self.net(xy)
        return z.cpu().numpy(), range(self.n_classes)

    def draw_region_img_3d(self, file_name="region_img_3d.png"):
        with self._ax(file_name, with_3d=True) as ax:
            plot_regions_3d(
                self.funcs,
                self.regions,
                z_fun=self._z_fun,
                ax=ax,
                alpha=0.8,
                color=color,
                edgecolor="grey",
                linewidth=0.2,
                xlim=self.bounds,
                ylim=self.bounds,
            )
            self._set_axis(ax)

    def draw_region_img_result(self, color_bar: bool = False, file_name: str = "region_img_result.png"):
        with self._ax(file_name) as ax:
            img = self.__draw_hot(ax)
            plot_regions(
                self.funcs,
                self.regions,
                ax=ax,
                color=lambda _: "w",
                alpha=0.1,
                edgecolor="black",
                linewidth=0.3,
                xlim=self.bounds,
                ylim=self.bounds,
            )
            if color_bar:
                ax.get_figure().colorbar(img, ax=ax)
            self._set_axis(ax)

    def __draw_hot(self, ax):
        num = 1000
        data = self.__hot_data(num).float()
        result = self.net(data).softmax(dim=1)
        result = (result - 1 / self.n_classes) / (1 - 1 / self.n_classes)
        result, maxIdx = torch.max(result, dim=1)
        result, maxIdx = result.cpu().numpy(), maxIdx.cpu().numpy()
        result_alpha, result_color = np.empty((num, num)), np.empty((num, num))
        for i in range(num):
            result_color[num - 1 - i] = maxIdx[i * num : (i + 1) * num]
            result_alpha[num - 1 - i] = result[i * num : (i + 1) * num]
        cmap = matplotlib.colors.ListedColormap(COLOR, name="Region")
        return ax.imshow(
            result_color,
            alpha=result_alpha,
            cmap=cmap,
            extent=(self.min_bound, self.max_bound, self.min_bound, self.max_bound),
            vmin=0,
            vmax=len(COLOR),
        )

    def __hot_data(self, num=1000):
        x1 = np.linspace(self.min_bound, self.max_bound, num)
        x2 = np.linspace(self.min_bound, self.max_bound, num)
        X1, X2 = np.meshgrid(x1, x2)
        X1, X2 = X1.flatten(), X2.flatten()
        data = np.vstack((X1, X2)).transpose()
        data = torch.from_numpy(data).to(self.device)
        return data

    def _path(self, file_name: str) -> str:
        return os.path.join(self.save_dir, file_name)

    def _ax(self, file_name: str, with_3d=False):
        return default_subplots(self._path(file_name), with_legend=False, with_grid=False, with_3d=with_3d)

    def _set_axis(self, ax: plt.Axes):
        if not self.with_ticks:
            ax.tick_params(which="both", colors="w")
