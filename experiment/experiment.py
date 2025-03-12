import json
import math
import os
from copy import deepcopy
from cProfile import label
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils import data

from dataset import Dataset
from torchays import nn
from torchays.cpa import CPA, Model, distance
from torchays.graph import bar, color, default_subplots
from torchays.utils import get_logger

from .draw import DrawRegionImage
from .handler import Handler
from .hpa import HyperplaneArrangement, HyperplaneArrangements


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


class _base:
    def __init__(
        self,
        save_dir: str,
        *,
        net: Callable[[int, bool], Model] = None,
        dataset: Callable[..., Tuple[Dataset, int]] = None,
        save_epoch: List[int] = [100],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.save_dir = save_dir
        self.net = net
        self.dataset = dataset
        self.save_epoch = save_epoch
        self.device = device
        self.training = True
        self.root_dir = None

    def _init_model(self):
        dataset, n_classes = self.dataset()
        net = self.net(n_classes, self.training).to(self.device)
        self._init_dir(net.name)
        return net, dataset, n_classes

    def _init_dir(self, name: str):
        self.root_dir = os.path.join(self.save_dir, name)
        self.model_dir = os.path.join(self.root_dir, "model")
        self.experiment_dir = os.path.join(self.root_dir, "experiment")
        for dir in [
            self.root_dir,
            self.model_dir,
            self.experiment_dir,
        ]:
            os.makedirs(dir, exist_ok=True)

    def val_net(self, net: nn.Module, val_dataloader: data.DataLoader) -> torch.Tensor:
        net.eval()
        val_accuracy_sum = 0
        for x, y in val_dataloader:
            x, y = x.float().to(self.device), y.long().to(self.device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
        return val_accuracy_sum

    def run(self):
        raise NotImplementedError()


class Train(_base):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        save_epoch: List[int] = [100],
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: Callable[[nn.Module, int, int, int, torch.Tensor, torch.Tensor, str], None] = None,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.train_handler = train_handler

    def run(self):
        net, dataset, _ = self._init_model()
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        total_step = math.ceil(len(dataset) / self.batch_size)

        optim = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0, betas=[0.9, 0.999])
        ce = torch.nn.CrossEntropyLoss()

        save_step = [v for v in self.save_epoch if v < 1]
        steps = [math.floor(v * total_step) for v in save_step]
        torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_0.pth'))
        best_acc, best_dict, best_epoch = 0, {}, 0
        for epoch in range(self.max_epoch):
            net.train()
            loss_sum = 0
            for j, (x, y) in enumerate(train_loader, 1):
                x: torch.Tensor = x.float().to(self.device)
                y: torch.Tensor = y.long().to(self.device)
                x = net(x)
                loss: torch.Tensor = ce(x, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                acc = accuracy(x, y) / x.size(0)
                loss_sum += loss

                if (epoch + 1) == 1 and (j in steps):
                    net.eval()
                    idx = steps.index(j)
                    torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{save_step[idx]}.pth'))
                    net.train()
                if self.train_handler is not None:
                    self.train_handler(net, epoch, j, total_step, loss, acc, self.model_dir)
                # print(f"Epoch: {epoch+1} / {self.max_epoch}, Step: {j} / {total_step}, Loss: {loss:.4f}, Acc: {acc:.4f}")
            net.eval()
            if (epoch + 1) in self.save_epoch:
                print(f"Save net: net_{epoch+1}.pth")
                torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{epoch+1}.pth'))
            with torch.no_grad():
                loss_sum = loss_sum / total_step
                acc = self.val_net(net, train_loader).cpu().numpy()
                print(f'Epoch: {epoch+1} / {self.max_epoch}, Loss: {loss_sum:.4f}, Accuracy: {acc:.4f}')
                if acc > best_acc:
                    best_acc, best_epoch = acc, epoch
                    best_dict = deepcopy(net.state_dict())
        torch.save(best_dict, os.path.join(self.model_dir, f'net_best_{best_epoch+1}.pth'))
        print(f'Best_Epoch: {best_epoch+1} / {self.max_epoch}, Accuracy: {best_acc:.4f}')


class _cpa(_base):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        save_epoch: List[int] = [100],
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.training = False
        self.bounds = bounds
        self.best_epoch = best_epoch
        self.depth = depth

    def _is_continue(self, model_name: str):
        epoch = float(model_name.split("_")[-1][:-4])
        if epoch in self.save_epoch:
            return False
        if self.best_epoch and "best" in model_name:
            return False
        return True

    def input_size(self, net: Model, dataset: Dataset) -> Tuple[int] | torch.Size:
        try:
            return net.input_size
        except:
            return dataset.input_size


class CPAs(_cpa):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        workers: int = 1,
        save_epoch: List[int] = [100],
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
        is_draw: bool = True,
        is_draw_3d: bool = False,
        is_draw_hpas: bool = False,
        is_statistic_hpas: bool = True,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(
            save_dir,
            net,
            dataset,
            save_epoch=save_epoch,
            best_epoch=best_epoch,
            bounds=bounds,
            depth=depth,
            device=device,
        )
        self.workers, self.multi = self._works(workers)
        self.is_draw = is_draw
        self.is_draw_3d = is_draw_3d
        self.is_draw_hpas = is_draw_hpas
        self.is_statistic_hpas = is_statistic_hpas
        self.is_hpas = is_draw_hpas or is_statistic_hpas

    def _works(self, workers: int):
        workers = math.ceil(workers)
        if workers <= 1:
            return 1, False
        return workers, True

    def run(self):
        net, dataset, n_classes = self._init_model()
        depth = self.depth if self.depth >= 0 else net.n_relu
        val_dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        input_size = self.input_size(net, dataset)
        cpa = CPA(device=self.device, workers=self.workers)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            print(f"Action: Net CPAs....")
            for model_name in model_list:
                if self._is_continue(model_name):
                    continue
                print(f"Solve fileName: {model_name} ....")
                save_dir = os.path.join(self.experiment_dir, os.path.splitext(model_name)[0])
                os.makedirs(save_dir, exist_ok=True)
                net.load_state_dict(torch.load(os.path.join(self.model_dir, model_name), weights_only=False))
                acc = self.val_net(net, val_dataloader).cpu().numpy()
                print(f"Accuracy: {acc:.4f}")
                handler = Handler() if self.is_draw or self.is_hpas else None
                logger = get_logger(
                    f"region-{os.path.splitext(model_name)[0]}",
                    os.path.join(save_dir, "region.log"),
                    multi=self.multi,
                )
                count = cpa.start(
                    net,
                    bounds=self.bounds,
                    input_size=input_size,
                    depth=depth,
                    handler=handler,
                    logger=logger,
                )
                print(f"Region counts: {count}")
                if self.is_draw:
                    draw_dir = os.path.join(save_dir, f"draw-region-{depth}")
                    os.makedirs(draw_dir, exist_ok=True)
                    dri = DrawRegionImage(count, handler.funs, handler.regions, handler.points, draw_dir, net, n_classes, bounds=self.bounds, device=self.device)
                    dri.draw(self.is_draw_3d)
                if self.is_hpas:
                    hpas = HyperplaneArrangements(save_dir, handler.hyperplane_arrangements, self.bounds)
                    hpas.run(is_draw=self.is_draw_hpas, is_statistic=self.is_statistic_hpas)
                data_dict = {
                    "funcs": handler.funs,
                    "regions": handler.regions,
                    "points": handler.points,
                    "regionNum": count,
                    "accuracy": acc,
                }
                torch.save(data_dict, os.path.join(save_dir, "net_regions.pkl"))
                result = {"regions": count, "accuracy": f"{acc:.4f}"}
                with open(os.path.join(save_dir, "results.json"), "w") as w:
                    json.dump(result, w)


class _distance:
    neural_ds: List[torch.Tensor]
    inter_ds: List[torch.Tensor]

    def __init__(self):
        self.inter_ds = list()
        self.neural_ds = list()

    def append(self, neural_ds: torch.Tensor = None, inter_ds: torch.Tensor = None):
        if neural_ds is not None:
            self.neural_ds.append(neural_ds)
        if inter_ds is not None:
            self.inter_ds.append(inter_ds)
        return self


class Points(_cpa):

    def __init__(
        self,
        save_dir: str,
        *,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
        save_epoch: List[int] = [100],
        best_epoch: bool = False,
        device=torch.device('cpu'),
    ):
        super().__init__(
            save_dir,
            net,
            dataset,
            save_epoch=save_epoch,
            best_epoch=best_epoch,
            bounds=bounds,
            depth=depth,
            device=device,
        )

    def run(self):
        net, dataset, _ = self._init_model()
        depth = self.depth if self.depth >= 0 else net.n_relu
        dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        input_size = self.input_size(net, dataset)
        cpa = CPA(device=self.device)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            # 在不同epoch下
            print(f"Action: Point Distance ....")
            for model_name in model_list:
                if self._is_continue(model_name):
                    continue
                print(f"Solve fileName: {model_name} ....")
                save_dir = os.path.join(self.experiment_dir, os.path.splitext(model_name)[0])
                os.makedirs(save_dir, exist_ok=True)
                net.load_state_dict(torch.load(os.path.join(self.model_dir, model_name), weights_only=False))
                logger = get_logger(f"region-{os.path.splitext(model_name)[0]}", os.path.join(save_dir, "points.log"))
                # 获取每个数据，在当前父区域下超平面的距离
                values: Dict[int, _distance] = dict()
                i = 0
                for point, _ in dataloader:
                    i += 1
                    point: torch.Tensor = point[0].float()
                    handler = Handler()
                    cpa.start(
                        net,
                        point,
                        bounds=self.bounds,
                        input_size=input_size,
                        depth=depth,
                        handler=handler,
                        logger=logger,
                    )
                    values = self._handler_hpas(values, point, handler.hyperplane_arrangements)
                draw_dir = os.path.join(save_dir, f"distance-count")
                os.makedirs(draw_dir, exist_ok=True)
                for depth, value in values.items():
                    save_path = os.path.join(draw_dir, f"distance-{depth}.png")
                    nd_x, nd_y = bar(value.neural_ds, 0.02)
                    id_x, id_y = bar(value.inter_ds, 0.02)
                    with default_subplots(save_path, "value", "count", with_grid=False, with_legend=False) as ax:
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(0, math.floor(sum(nd_y) / 5))
                        ax.bar(nd_x, nd_y, color=color(1), width=0.05, label=f"All Neurons: {sum(nd_y)}")
                        ax.bar(id_x, id_y, color=color(0), width=0.05, label=f"Intersect Neurons: {sum(id_y)}")
                        ax.legend(prop={"weight": "normal", "size": 7})

    def _handler_hpas(
        self,
        values: Dict[int, _distance],
        point: torch.Tensor,
        hpas: Dict[int, List[HyperplaneArrangement]],
    ):
        for depth, hpa in hpas.items():
            hpa = hpa.pop()
            nerual_ds = distance(point, hpa.c_funs)
            inter_ds = None
            if hpa.intersect_funs is not None:
                inter_ds = distance(point, hpa.intersect_funs)
            values[depth] = values.pop(depth, _distance()).append(nerual_ds, inter_ds)
        return values


class Experiment(_base):
    def __init__(
        self,
        save_dir: str,
        init_fun: Callable[..., None],
        *,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        save_epoch: List[int] = [100],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.init_fun = init_fun
        self.runs = list()

    def train(
        self,
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: Callable[[nn.Module, int, int, int, torch.Tensor, torch.Tensor, str], None] = None,
    ):
        train = Train(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            max_epoch=max_epoch,
            batch_size=batch_size,
            lr=lr,
            train_handler=train_handler,
            device=self.device,
        )
        self.append(train.run)

    def cpas(
        self,
        workers: int = 1,
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
        is_draw: bool = False,
        is_draw_3d: bool = False,
        is_draw_hpas: bool = False,
        is_statistic_hpas: bool = False,
    ):
        cpas = CPAs(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            workers=workers,
            best_epoch=best_epoch,
            bounds=bounds,
            depth=depth,
            is_draw=is_draw,
            is_draw_3d=is_draw_3d,
            is_draw_hpas=is_draw_hpas,
            is_statistic_hpas=is_statistic_hpas,
            device=self.device,
        )
        self.append(cpas.run)

    def points(
        self,
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
    ):
        points = Points(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            best_epoch=best_epoch,
            bounds=bounds,
            depth=depth,
            device=self.device,
        )
        self.append(points.run)

    def append(self, fun: Callable[..., None]):
        self.runs.append(self.init_fun)
        self.runs.append(fun)

    def run(self):
        for run in self.runs:
            run()

    def __call__(self, *args: Any, **kwds: Any):
        self.run(*args, **kwds)
