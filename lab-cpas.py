import os
from typing import Dict, List

import torch

from config import EXPERIMENT, GLOBAL, PATH, TESTNET, TOY, TRAIN
from experiment.draw import bar
from run import dataset, net, proj_net, run
from torchays import nn
from torchays.graph import color, default_subplots


class Handler:
    def __init__(self, root_dir: str = "./"):
        self.epoch, self.step = -1, -1
        self.batch_norm_data: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = dict()
        self.norm_neural_value: Dict[int, Dict[str, List[torch.Tensor]]] = dict()
        self.root_dir = os.path.join(root_dir, "neural_regions")

    def train_handler(
        self,
        net: nn.Module,
        epoch: int,
        step: int,
        total_step: int,
        loss: torch.Tensor,
        acc: torch.Tensor,
        save_dir: str,
    ):
        self.epoch, self.step = epoch, step
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
            # 计算对应的A_bn和B_bn
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
        self.batch_norm_data[step_name] = current_bn_data

    def net_handler(self, key: str, value: torch.Tensor, description: str = ""):
        # TODO: 有问题
        epoch = self.epoch + 1
        if epoch not in TRAIN.SAVE_EPOCH:
            return
        norm_v = self.norm_neural_value.pop(epoch, dict())
        v_list = norm_v.pop(key, list())
        v_list.append(value.clone().detach().cpu())
        norm_v[key] = v_list
        self.norm_neural_value[epoch] = norm_v

    def save(self, name: str = "norm.pkl"):
        os.makedirs(self.root_dir, exist_ok=True)
        save_dict = {
            "norm_values": self.norm_neural_value,
            "data": self.batch_norm_data,
        }
        torch.save(save_dict, os.path.join(self.root_dir, name))

    def statistic(
        self,
        name: str = None,
        with_values: bool = True,
        with_data: bool = True,
    ):
        if name is not None:
            save_dict: Dict[str, Dict] = torch.load(os.path.join(self.root_dir, name), weights_only=False)
            self.norm_neural_value = save_dict.get("norm_values")
            self.batch_norm_data = save_dict.get("data")
        if with_values:
            self._statistic_values()
        if with_data:
            self._statistic_data()

    def _statistic_values(self):
        # {epoch:{k1:[v1, v2...]}}
        # 绘制每一层，神经元值的分布图。
        # 以及平均分布图
        save_dir = os.path.join(self.root_dir, "values")
        os.makedirs(save_dir, exist_ok=True)
        for epoch, kvs in self.norm_neural_value.items():
            # 先画每个epoch的,所有值的分布图
            for k, v in kvs.items():
                save_path = os.path.join(save_dir, f"epoch_{epoch}-{k}.png")
                # k是每一层
                # v是这一层每个step的值
                bar_x, bar_y = bar(v, 0.2, self._value)
                with default_subplots(save_path, "value", "log_prob", with_grid=False, with_legend=False) as ax:
                    ax.set_ylabel("counts", fontdict={"weight": "normal", "size": 15})
                    ax.bar(bar_x, bar_y, color=color(1), width=0.15, label=f"All Neurons: {sum(bar_y)}")
                    ax.legend(prop={"weight": "normal", "size": 7})

    def _value(self, v: torch.Tensor) -> torch.Tensor:
        return torch.log(v.abs())

    def _statistic_data(self):
        norm_data = self.batch_norm_data
        save_dict: Dict[str, Dict[int, Dict[str, torch.Tensor]] | List[str]] = dict()
        step_list = list(norm_data.keys())
        steps = len(step_list)
        name_list = ["weight", "bias", "running_mean", "running_var", "weight_bn", "bias_bn"]
        for j in range(steps):
            step_name = step_list[j]
            step_data = norm_data.pop(step_name)
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


def set_config(
    name: str,
    type: str,
    norm_layer,
):
    GLOBAL.NAME = name
    GLOBAL.TYPE = type
    TESTNET.NORM_LAYER = norm_layer


def config():
    PATH.TAG = "cpas-norm"

    TOY.N_SAMPLES = 500
    TOY.N_CLASS = 3

    TRAIN.TRAIN = True
    TRAIN.MAX_EPOCH = 5000
    TRAIN.SAVE_EPOCH = [1, 5, 10, 50, 100, 500, 1000, 1500, 2000, 3000, 5000]

    TESTNET.N_LAYERS = [64] * 5

    EXPERIMENT.CPAS = True
    EXPERIMENT.POINT = True
    EXPERIMENT.WORKERS = 64


def main():
    data = dataset()
    handler = Handler(os.path.join(data.path, GLOBAL.NAME))
    run(
        dataset=data,
        net=proj_net(
            net(),
            proj_dims=EXPERIMENT.PROJ_DIM,
            proj_values=EXPERIMENT.PROJ_VALUES,
            handler=handler.net_handler,
        ),
        train_handler=handler.train_handler,
    )
    if TRAIN.TRAIN:
        handler.save("norm.pkl")
    # handler.statistic("norm.pkl", with_data=False)


if __name__ == "__main__":
    from config import Classification, GaussianQuantiles, Moon

    configs = [
        # ("Linear-[64]x5-norm", "Random", nn.Norm1d),
        # ("Linear-[64]x5-batch", "Random", nn.BatchNorm1d),
        # ("Linear-[64]x5-norm", "Moon", nn.Norm1d),
        # ("Linear-[64]x5-batch", "Moon", nn.BatchNorm1d),
        # ("Linear-[32]x3-batch", "Moon", nn.Norm1d),
        ("Linear-[64]x5-norm", GaussianQuantiles, nn.Norm1d),
        ("Linear-[64]x5-batch", GaussianQuantiles, nn.BatchNorm1d),
    ]
    for cfg in configs:
        config()
        set_config(*cfg)
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")
