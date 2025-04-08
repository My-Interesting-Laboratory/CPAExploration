import os
from typing import Callable, Dict, List

import torch
from torch.nn.parameter import Parameter

from config import EXPERIMENT, GLOBAL, PATH, TESTNET, TOY, TRAIN
from experiment import TrainHandler
from experiment.draw import bar
from run import dataset, net, proj_net, run
from torchays import nn
from torchays.graph import color, default_subplots


def value(v: torch.Tensor) -> torch.Tensor:
    return torch.log(v.abs())


class Handler(TrainHandler):
    def __init__(self, root_dir: str = "./", epoch_transformer: Callable[[nn.Module, int], None] | None = None):
        self.epoch = -1
        self.epoch_transformer = epoch_transformer
        self.step_data: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = dict()
        self.epoch_data: Dict[int, Dict[str, torch.Tensor]] = dict()
        self.net_data: Dict[int, Dict[str, List[torch.Tensor]]] = dict()
        self.root_dir = os.path.join(root_dir, "handlers")

    def step_handler(
        self,
        net: nn.Module,
        epoch: int,
        step: int,
        total_step: int,
        loss: torch.Tensor,
        acc: torch.Tensor,
    ):
        step_name = f"{epoch}/{step}"
        current_bn_data = dict()
        for layer_name, module in net._modules.items():
            if "_norm" not in layer_name or not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            module: nn.BatchNorm1d
            parameters: Dict[str, torch.Tensor] = module.state_dict()
            weight = parameters.get("weight").cpu()
            bias = parameters.get("bias").cpu()
            running_mean = parameters.get("running_mean").cpu()
            running_var = parameters.get("running_var").cpu()
            num_batches_tracked = parameters.get("num_batches_tracked").cpu()
            p = torch.sqrt(running_var + module.eps)
            # weight_bn = w/√(var)
            weight_bn = weight / p
            # bias_bn = b - w*mean/√(var)
            bias_bn = bias - weight_bn * running_mean
            save_dict = {
                "weight": weight,
                "bias": bias,
                "running_mean": running_mean,
                "running_var": running_var,
                "num_batches_tracked": num_batches_tracked,
                "weight_bn": weight_bn,
                "bias_bn": bias_bn,
            }
            current_bn_data[layer_name] = save_dict
        self.step_data[step_name] = current_bn_data

    def epoch_handler(self, net: nn.Module, epoch: int, loss: torch.Tensor, acc: float):
        self.epoch = epoch + 1
        self.epoch_data[epoch] = {
            "accuracy": acc,
            "loss": loss,
        }
        if self.epoch_transformer is not None:
            self.epoch_transformer(net, epoch)

    def net_handler(self, key: str, value: torch.Tensor, description: str = ""):
        epoch = self.epoch + 1
        if epoch not in TRAIN.SAVE_EPOCH:
            return
        norm_v = self.net_data.pop(epoch, dict())
        v_list = norm_v.pop(key, list())
        v_list.append(value.clone().detach().cpu())
        norm_v[key] = v_list
        self.net_data[epoch] = norm_v

    def save(self, name: str = "handler.pkl"):
        os.makedirs(self.root_dir, exist_ok=True)
        save_dict = {
            "step_data": self.step_data,
            "epoch_data": self.epoch_data,
            "net_data": self.net_data,
        }
        torch.save(save_dict, os.path.join(self.root_dir, name))

    def statistic(
        self,
        name: str = None,
        with_net_data: bool = True,
        with_step_data: bool = True,
        with_epoch_data: bool = True,
    ):
        if not with_net_data and not with_step_data and not with_epoch_data:
            return
        if name is not None:
            save_dict: Dict[str, Dict] = torch.load(os.path.join(self.root_dir, name), weights_only=False)
            self.step_data = save_dict.get("step_data")
            self.epoch_data = save_dict.get("epoch_data")
            self.net_data = save_dict.get("net_data")
        if with_net_data:
            self._statistic_net_data()
        if with_step_data:
            self._statistic_step_data()
        if with_epoch_data:
            self._statistic_epoch_data()

    def _statistic_net_data(self):
        # {epoch:{k1:[v1, v2...]}}
        save_dir = os.path.join(self.root_dir, "values")
        os.makedirs(save_dir, exist_ok=True)
        for epoch, kvs in self.net_data.items():
            for k, v in kvs.items():
                save_path = os.path.join(save_dir, f"epoch_{epoch}-{k}.jpg")
                bar_x, bar_y = bar(v, 0.2, value)
                with default_subplots(save_path, "value", "log_prob", with_grid=False, with_legend=False) as ax:
                    ax.set_ylabel("counts", fontdict={"weight": "normal", "size": 15})
                    ax.bar(bar_x, bar_y, color=color(1), width=0.15, label=f"All Neurons: {sum(bar_y)}")
                    ax.legend(prop={"weight": "normal", "size": 7})

    def _statistic_step_data(self):
        step_data = self.step_data
        save_dict: Dict[str, Dict[int, Dict[str, torch.Tensor]] | List[str]] = dict()
        step_list = list(step_data.keys())
        steps = len(step_list)
        name_list = ["weight", "bias", "running_mean", "running_var", "weight_bn", "bias_bn"]
        for j in range(steps):
            step_name = step_list[j]
            step_data = step_data.pop(step_name)
            for layer_name, layer_data in step_data.items():
                for name in name_list:
                    data = layer_data.pop(name)
                    for i in range(len(data)):
                        neurons = save_dict.pop(layer_name, dict())
                        values = neurons.pop(i, dict())
                        value = values.pop(name, torch.zeros(steps))
                        value[j] = data[i]
                        values[name] = value
                        neurons[i] = values
                        save_dict[layer_name] = neurons
        save_dict["steps"] = step_list
        self._draw_parameters(save_dict)

    def _draw_parameters(self, save_dict: Dict[str, Dict[int, Dict[str, torch.Tensor]]]):
        save_dir = os.path.join(self.root_dir, "parameters")
        os.makedirs(save_dir, exist_ok=True)
        step_list = save_dict.pop("steps", list())
        for layer_name, neurons in save_dict.items():
            for j, values in neurons.items():
                layer_dir = os.path.join(save_dir, layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                save_path = os.path.join(layer_dir, f"neuron_{j}.png")
                with default_subplots(save_path, "steps", "values") as ax:
                    i = 0
                    for name, value in values.items():
                        ax.plot(range(len(step_list)), value, label=name, color=color(i))
                        i += 1

    def _statistic_epoch_data(self):
        epoch_data = self.epoch_data
        accs = torch.zeros(len(epoch_data))
        for epoch, data in epoch_data.items():
            acc = data.get("accuracy", 0)
            accs[epoch] = acc
        rating = accs

        save_dir = os.path.join(self.root_dir, "epoch_data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "accuray.jpg")
        with default_subplots(save_path, "Epoch", "Accuracy", with_grid=False, with_legend=False) as ax:
            ax.plot(range(len(rating)), rating, label="Accuracy", color=color(0))


class AnalysisData:
    def __init__(self, name: str):
        self._xs = []
        self.ys = []
        self.name = name

    def append(self, y, x=None):
        self.ys.append(y)
        if x is not None:
            self._xs.append(x)

    @property
    def xs(self):
        if len(self._xs) != len(self.ys):
            return range(len(self.ys))
        return self._xs


class Analysis:
    def __init__(self, root: str, handler_name: str = "handler.pkl"):
        self.root = root
        self.save_dir = os.path.join(root, "norm-analysis")
        os.makedirs(self.save_dir, exist_ok=True)
        self.handler_name = handler_name

    def run(self, with_handlers: bool = True, with_net_regions: bool = False):
        if with_handlers:
            self._handlers()
        if with_net_regions:
            self._net_regions()

    def _handlers(self):
        """
        name/handlers/handlers.pkl
        {
            "step_data": self.step_data,
            "epoch_data": self.epoch_data,
            "net_data": self.net_data,
        }
        """
        names = os.listdir(self.root)
        handlers: Dict[str, List[AnalysisData]] = {}
        for name in names:
            norm_file = os.path.join(self.root, name, "handlers", self.handler_name)
            if not os.path.exists(norm_file):
                continue
            data: Dict[str, Dict] = torch.load(norm_file, weights_only=False)
            epoch_data = data.get("epoch_data", {})
            acc, loss = AnalysisData(name), AnalysisData(name)
            for epoch, data in epoch_data.items():
                acc.append(data.get("accuracy", torch.zeros(1)).item(), epoch)
                loss.append(data.get("loss", torch.zeros(1)).item(), epoch)
            al, ll = handlers.pop("Accuracy", list()), handlers.pop("Loss", list())
            al.append(acc)
            ll.append(loss)
            handlers["Accuracy"], handlers["Loss"] = al, ll

        self._draw_data(handlers)

    def _net_regions(self):
        """
        name/experiment/net_*/net_regions.pkl
        {
            "funcs": handler.funs,
            "regions": handler.regions,
            "points": handler.points,
            "regionNum": count,
            "accuracy": acc,
        }
        """
        names = os.listdir(self.root)
        nr_handlers: Dict[str, List[AnalysisData]] = {}
        for name in names:
            exp_dir = os.path.join(self.root, name, "experiment")
            if not os.path.exists(exp_dir) or not os.path.isdir(exp_dir):
                continue
            regions = AnalysisData(name)
            epoch_dir = os.listdir(exp_dir)
            epoch_dir.sort(key=lambda elem: int(elem.split("_")[-1]))
            for epoch_dir in epoch_dir:
                epoch = int(epoch_dir.split("_")[-1])
                nr_path = os.path.join(exp_dir, epoch_dir, 'net_regions.pkl')
                if not os.path.isfile(nr_path):
                    continue
                data: Dict[str, int] = torch.load(nr_path, weights_only=False)
                regions.append(data.get("regionNum", 0), epoch)
            rl = nr_handlers.pop("Regions", list())
            rl.append(regions)
            nr_handlers["Regions"] = rl
        self._draw_data(nr_handlers)

    def _draw_data(self, data: Dict[str, List[AnalysisData]]):
        for quota, handler in data.items():
            save_path = os.path.join(self.save_dir, f"{quota}.jpg")
            with default_subplots(save_path, "Epoch", quota, with_grid=False, with_legend=False) as ax:
                for i, ad in enumerate(handler):
                    ax.plot(ad.xs, ad.ys, label=ad.name, color=color(i), alpha=0.7)
                ax.legend(prop={"weight": "normal", "size": 7})


def main():
    data = dataset()
    handler = Handler(os.path.join(data.path, GLOBAL.NAME), HandlerConfig.EPOCH_TRANSFORMER)
    run(
        dataset=data,
        net=proj_net(
            net(),
            proj_dims=EXPERIMENT.PROJ_DIM,
            proj_values=EXPERIMENT.PROJ_VALUES,
            handler=handler.net_handler,
        ),
        train_handler=handler,
    )
    if TRAIN.TRAIN:
        handler.save("handler.pkl")
    handler.statistic("handler.pkl", HandlerConfig.WITH_NET_DATA, HandlerConfig.WITH_STEP_DATA, HandlerConfig.WITH_EPOCH_DATA)


def epoch_transformer(callable: Callable[[nn.BatchNorm1d], nn.BatchNorm1d | nn.Norm1d], info: str):
    def transformer(net: nn.Module, epoch: int):
        if epoch != 100:
            return
        print(f"Change the \"BatchNorm\" to \"{info}\"")
        norm_dict = {}
        for k, v in net._modules.items():
            if "_norm" not in k:
                continue
            v: nn.BatchNorm1d
            norm = callable(v).to(v.weight.device)
            norm_dict[k] = norm
        net._modules.update(norm_dict)

    return transformer


def epoch_linear():
    def callable(v: nn.BatchNorm1d) -> nn.Norm1d:
        def _set_parameters(weight: Parameter, bias: Parameter):
            p = torch.sqrt(v.running_var + v.eps)
            # weight_bn = w/√(var)
            weight_bn = v.weight / p
            # bias_bn = b - w*mean/√(var)
            bias_bn = v.bias - weight_bn * v.running_mean

            weight.copy_(weight_bn)
            bias.copy_(bias_bn)

        return nn.Norm1d(v.num_features, False, _set_parameters)

    return epoch_transformer(callable, "Linear")


class HandlerConfig:
    WITH_NET_DATA: bool = True
    WITH_STEP_DATA: bool = True
    WITH_EPOCH_DATA: bool = True
    EPOCH_TRANSFORMER: Callable[[nn.Module, int], None] | None = None

    ANALYSIS: bool = True
    WITH_HANDLERS: bool = True
    WITH_NET_REGIONS: bool = False


def set_config(
    name: str,
    type: str,
    norm_layer,
    epoch_transformer: Callable[[nn.Module, int], None] | None = None,
):
    GLOBAL.NAME = name
    GLOBAL.TYPE = type
    TESTNET.NORM_LAYER = norm_layer
    HandlerConfig.EPOCH_TRANSFORMER = epoch_transformer


def config():
    PATH.TAG = "cpas-norm"

    TOY.N_SAMPLES = 500
    TOY.N_CLASSES = 4
    TOY.IN_FEATURES = 2

    TRAIN.TRAIN = False
    TRAIN.MAX_EPOCH = 30000
    TRAIN.SAVE_EPOCH = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 30000]
    TRAIN.BATCH_SIZE = 256
    TRAIN.LR = 5e-4

    TESTNET.N_LAYERS = [32] * 5

    EXPERIMENT.WORKERS = 64
    EXPERIMENT.CPAS = False
    EXPERIMENT.POINT = False

    HandlerConfig.WITH_NET_DATA = False
    HandlerConfig.WITH_STEP_DATA = False
    HandlerConfig.WITH_EPOCH_DATA = False
    HandlerConfig.ANALYSIS = True


if __name__ == "__main__":
    from config import Classification, GaussianQuantiles, Moon, Random

    configs = [
        ("Linear-[32]x5-norm", Random, nn.Norm1d),
        ("Linear-[32]x5-batch", Random, nn.BatchNorm1d),
        ("Linear-[32]x5-batch-linear", Random, nn.BatchNorm1d, epoch_linear()),
    ]

    for cfg in configs:
        config()
        set_config(*cfg)
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")

    if HandlerConfig.ANALYSIS:
        print("========= Now: Analysis =========")
        config()
        set_config(*configs[0])
        analysis = Analysis(dataset().path)
        analysis.run(HandlerConfig.WITH_HANDLERS, HandlerConfig.WITH_NET_REGIONS)
        print("========= End: Analysis =========")
