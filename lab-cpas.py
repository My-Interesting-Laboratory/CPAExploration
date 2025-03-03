import os
from typing import Any, Dict, List, Tuple

import torch

from config import EXPERIMENT, GLOBAL, TRAIN
from main import main, proj_net
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
            if "_norm" not in layer_name:
                continue
            # 存储每一个batch下的bn的参数
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
        epoch = self.epoch + 1
        if epoch not in TRAIN.SAVE_EPOCH:
            return
        norm_v = self.norm_neural_value.pop(epoch, dict())
        v_list = norm_v.pop(key, list())
        v_list.append(value)
        norm_v[key] = v_list
        norm_v[epoch] = norm_v

    def save(self, name: str = "norm.pkl"):
        os.makedirs(self.root_dir)
        self.norm_neural_value
        self.batch_norm_data
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

    def _statistic_data(self):
        data = self.batch_norm_data
        save_dict: Dict[str, Dict[int, Dict[str, torch.Tensor]] | List[str]] = dict()
        step_list = list(data.keys())
        steps = len(step_list)
        name_list = ["weight", "bias", "running_mean", "running_var", "weight_bn", "bias_bn"]
        for j in range(steps):
            step_name = step_list[j]
            step_data = data.pop(step_name)
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

    def _statistic_values(self):
        # {epoch:{k1:[v1, v2...]}}
        # 绘制每一层，神经元值的分布图。
        # 以及平均分布图
        for epoch, kvs in self.norm_neural_value.items():
            save_dir = os.path.join(self.root_dir, "values")
            # 先画每个epoch的,所有值的分布图
            for k, v in kvs.items():
                save_path = os.path.join(save_dir, f"epoch_{epoch}_{k}.png")
                # k是每一层
                # v是这一层每个step的值
                values, probs = self._values(v)
                with default_subplots(save_path, "value", "prob", with_grid=False, with_legend=False) as ax:
                    ax.plot(values, probs, color=color(0))

    def _values(self, values: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 先求这个数据的高斯分布, 然后画出高斯分布后, 根据值计算其概率，绘制概率分布
        values = torch.cat(values).reshape(-1)
        std, mean = torch.std_mean(values)
        gaussian = torch.distributions.Normal(mean, std)
        probs = gaussian.log_prob(values)
        return values, probs


if __name__ == "__main__":
    dataset = GLOBAL.dataset()
    handler = Handler(dataset.path)
    main(
        dataset=GLOBAL.dataset(),
        net=proj_net(
            GLOBAL.net(),
            proj_dims=EXPERIMENT.PROJ_DIM,
            proj_values=EXPERIMENT.PROJ_VALUES,
            handler=handler.net_handler,
        ),
        train_handler=handler.train_handler,
    )
    handler.save("norm.pkl")
    handler.statistic()
