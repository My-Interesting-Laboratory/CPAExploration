import json
import math
import os
import time
from copy import deepcopy
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils import data

from torchays import nn
from torchays.cpa import CPAFactory, Model, ProjectWrapper, distance
from torchays.utils import get_logger

from .draw import DrawRegionImage
from .handler import Handler, TrainHandler
from .hpa import HyperplaneArrangement, HyperplaneArrangements
from .point import Neurals


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
        dataset: Callable[..., Tuple[data.Dataset, int]] = None,
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
        self._init_dir(net)
        return net, dataset, n_classes

    def _init_dir(self, net: Model):
        self.root_dir = os.path.join(self.save_dir, net.name)
        self.model_dir = os.path.join(self.root_dir, "model")
        exp_dir = "experiment"
        if isinstance(net, ProjectWrapper) and net.proj_dim is not None:
            exp_dir = f"experiment-{net.proj_dim}".replace(" ", "")
        self.experiment_dir = os.path.join(self.root_dir, exp_dir)
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
        dataset: Callable[..., Tuple[data.Dataset, int]],
        *,
        save_epoch: List[int] = [100],
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: TrainHandler = None,
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
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=True, pin_memory=True)
        total_step = math.ceil(len(dataset) / self.batch_size)

        optim = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0, betas=[0.9, 0.999])
        ce = torch.nn.CrossEntropyLoss()

        save_step = [v for v in self.save_epoch if v < 1]
        steps = [math.floor(v * total_step) for v in save_step]
        torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_0.pth'))
        best_acc, best_dict, best_epoch = 0, {}, 0
        for epoch in range(self.max_epoch):
            net.train()
            t_start = time.time()
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
                    self.train_handler.step_handler(net, epoch, j, total_step, loss, acc)
                # print(f"Epoch: {epoch+1} / {self.max_epoch}, Step: {j} / {total_step}, Loss: {loss:.4f}, Acc: {acc:.4f}")
            net.eval()
            if (epoch + 1) in self.save_epoch:
                print(f"Save net: net_{epoch+1}.pth")
                torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{epoch+1}.pth'))
            with torch.no_grad():
                loss_avg = loss_sum / total_step
                acc = self.val_net(net, train_loader).cpu()
                if self.train_handler is not None:
                    self.train_handler.epoch_handler(net, epoch, loss_avg, acc)
                print(f"Epoch: {epoch+1} / {self.max_epoch}, Loss: {loss_avg:.4f}, Accuracy: {acc:.4f}, Time: {time.time()-t_start:.3f}s")
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
        dataset: Callable[..., Tuple[data.Dataset, int]],
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
        if "best" in model_name:
            return not self.best_epoch
        epoch = float(model_name.split("_")[-1][:-4])
        if epoch in self.save_epoch:
            return False
        return True

    def input_size(self, net: Model, dataset: data.Dataset) -> Tuple[int] | torch.Size:
        try:
            return net.input_size
        except:
            return dataset.input_size


class CPAs(_cpa):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[data.Dataset, int]],
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
        depth = self.depth if self.depth >= 0 else net.depth
        val_dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        input_size = self.input_size(net, dataset)
        cpa_factory = CPAFactory(device=self.device, workers=self.workers)
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
                net.to(self.device)
                # Accuracy
                acc = self.val_net(net, val_dataloader).cpu().numpy()
                print(f"Accuracy: {acc:.4f}")
                # CPA
                handler = Handler(self.is_draw, self.is_hpas)
                logger = get_logger(
                    f"{net.name}-{randint(-100,100)}-region_{os.path.splitext(model_name)[0]}",
                    os.path.join(save_dir, "region.log"),
                    multi=self.multi,
                )
                cpa = cpa_factory.CPA(net=net, depth=depth, handler=handler, logger=logger)
                count = cpa.start(input_size=input_size, bounds=self.bounds)
                print(f"Region counts: {count}")
                if self.is_draw:
                    draw_dir = os.path.join(save_dir, f"draw-region-{depth}")
                    os.makedirs(draw_dir, exist_ok=True)
                    dri = DrawRegionImage(
                        count,
                        handler.funs,
                        handler.regions,
                        handler.points,
                        draw_dir,
                        net,
                        n_classes,
                        bounds=self.bounds,
                        device=self.device,
                        with_ticks=False,
                    )
                    dri.draw(self.is_draw_3d)
                if self.is_hpas:
                    hpas = HyperplaneArrangements(save_dir, handler.hyperplane_arrangements, self.bounds)
                    hpas.run(is_draw=self.is_draw_hpas, is_statistic=self.is_statistic_hpas)
                result = {"regions": count, "accuracy": f"{acc:.4f}"}
                with open(os.path.join(save_dir, "results.json"), "w") as w:
                    json.dump(result, w)
                if handler is not None:
                    data_dict = {
                        "funcs": handler.funs,
                        "regions": handler.regions,
                        "points": handler.points,
                        "regionNum": count,
                        "accuracy": acc,
                    }
                    torch.save(data_dict, os.path.join(save_dir, "net_regions.pkl"))


class Points(_cpa):

    def __init__(
        self,
        save_dir: str,
        *,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[data.Dataset, int]],
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
        depth = self.depth if self.depth >= 0 else net.depth
        dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        cpa_factory = CPAFactory(device=self.device)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            print(f"Action: Point Distance ....")
            for model_name in model_list:
                if self._is_continue(model_name):
                    continue
                print(f"Solve fileName: {model_name} ....")
                save_dir = os.path.join(self.experiment_dir, os.path.splitext(model_name)[0])
                os.makedirs(save_dir, exist_ok=True)
                net.load_state_dict(torch.load(os.path.join(self.model_dir, model_name), weights_only=False))
                net.to(self.device)
                logger = get_logger(f"region-{os.path.splitext(model_name)[0]}", os.path.join(save_dir, "points.log"))
                values: Dict[int, Neurals] = dict()
                i = 0
                for point, _ in dataloader:
                    i += 1
                    point: torch.Tensor = point[0].float()
                    handler = Handler(False, True)
                    cpa = cpa_factory.CPA(net=net, depth=depth, handler=handler, logger=logger)
                    cpa.start(point=point, bounds=self.bounds)
                    values = self._handler_hpas(values, point, handler.hyperplane_arrangements)
                draw_dir = os.path.join(save_dir, f"distance-count")
                os.makedirs(draw_dir, exist_ok=True)
                for depth, value in values.items():
                    value.draw_bar(draw_dir, depth)

    def _handler_hpas(
        self,
        values: Dict[int, Neurals],
        point: torch.Tensor,
        hpas: Dict[int, List[HyperplaneArrangement]],
    ):
        for depth, hpa in hpas.items():
            hpa = hpa.pop()
            nerual_ds, nerual_v, nerual_ws = distance(point, hpa.c_funs)
            inter_ds, inter_v, inter_ws = None, None, None
            if hpa.intersect_funs is not None:
                inter_ds, inter_v, inter_ws = distance(point, hpa.intersect_funs)
            data_map = {
                "distance": (nerual_ds, inter_ds),
                "values": (nerual_v, inter_v),
                "weights": (nerual_ws, inter_ws),
            }
            values[depth] = values.pop(depth, Neurals(*data_map.keys())).append(data_map)
        return values


class Experiment(_base):
    def __init__(
        self,
        save_dir: str,
        init_fun: Callable[..., None],
        *,
        net: Callable[[int, bool], Model],
        dataset: Callable[..., Tuple[data.Dataset, int]],
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

    def _info(self):
        net, dataset, n_classes = self._init_model()
        parameters = sum([param.nelement() for param in net.parameters()])
        result = {
            "Net": {
                "n_parameters": parameters,
            },
            "data.Dataset": {
                "n_dataset": len(dataset),
                "n_classes": n_classes,
            },
        }
        with open(os.path.join(self.root_dir, "info.json"), "w") as w:
            json.dump(result, w)

    def train(
        self,
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: TrainHandler = None,
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
        self._info()
        for run in self.runs:
            run()

    def __call__(self, *args: Any, **kwds: Any):
        self.run(*args, **kwds)
