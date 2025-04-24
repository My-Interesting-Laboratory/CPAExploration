from random import Random
from typing import Any, Callable, List, Tuple

import torch

from config import ANALYSIS, CIFAR10, COMMON, EXPERIMENT, GLOBAL, MNIST, PATH, TESTNET, TOY, TRAIN, TYPE
from run import dataset, net, proj_net, run
from torchays import nn


def config():
    COMMON.GPU_ID = 7
    TRAIN.TRAIN = False
    TRAIN.MAX_EPOCH = 1000
    TRAIN.LR = 1e-3

    EXPERIMENT.CPAS = False
    EXPERIMENT.WORKERS = 64
    EXPERIMENT.BOUND = (-1, 1)

    EXPERIMENT.WITH_DRAW = True
    EXPERIMENT.WITH_DRAW_3D = False
    EXPERIMENT.WITH_DRAW = False

    ANALYSIS.WITH_DATASET = False


def set_config(
    name: str,
    type: str,
    batch_size: int = 64,
):
    GLOBAL.NAME = name
    GLOBAL.TYPE = type
    TRAIN.BATCH_SIZE = batch_size
    config()


def lab1(
    name: str,
    type: str,
    n_layers: List[int],
    norm_layer,
    in_features: int = 2,
    n_classes: int = 2,
):
    args = (name, type)
    kwargs = {}

    def config_fn():
        PATH.TAG = "CPAExploration-lab1"
        set_config(*args, **kwargs)

        TESTNET.N_LAYERS = n_layers
        TESTNET.NORM_LAYER = norm_layer

        TOY.N_SAMPLES = 200
        TOY.IN_FEATURES = in_features
        TOY.N_CLASSES = n_classes

        EXPERIMENT.WITH_DRAW_3D = True
        EXPERIMENT.WITH_DRAW = False if in_features > 2 else True
        ANALYSIS.WITH_DATASET = False if in_features > 2 else True

    return config_fn


def lab2(
    name: str,
    type: str,
    n_layers: List[int],
    norm_layer,
    in_features: int = 2,
):
    args = (name, type)
    kwargs = {}

    def config_fn():
        PATH.TAG = "CPAExploration-lab2"
        set_config(*args, **kwargs)
        TOY.IN_FEATURES = in_features
        TOY.N_SAMPLES = 200

        TESTNET.N_LAYERS = n_layers
        TESTNET.NORM_LAYER = norm_layer

        EXPERIMENT.WITH_DRAW_3D = True
        EXPERIMENT.WITH_DRAW_HPAS = True
        EXPERIMENT.WITH_STATISTIC_HPAS = True

        ANALYSIS.WITH_DATASET = True

    return config_fn


def lab3(
    name: str,
    type: str,
    norm_layer,
    linear: bool,
    batch_size: int = 64,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    args = (name, type)
    kwargs = {"batch_size": batch_size}

    def config_fn():
        PATH.TAG = "CPAExploration-lab3"
        set_config(*args, **kwargs)
        CIFAR10.NORM_LAYER = norm_layer
        CIFAR10.LINEAR = linear
        EXPERIMENT.PROJ_DIM = proj_dim
        EXPERIMENT.PROJ_VALUES = proj_values

    return config_fn


def lab4(
    name: str,
    type: str,
    norm_layer,
    n_layers: List[int] = None,
    batch_size: int = 64,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    args = (name, type)
    kwargs = {"batch_size": batch_size}

    def config_fn():
        PATH.TAG = "CPAExploration-lab4"
        set_config(*args, **kwargs)
        TESTNET.N_LAYERS = n_layers
        TESTNET.NORM_LAYER = norm_layer

        TOY.N_SAMPLES = 200

        CIFAR10.NORM_LAYER = norm_layer

        if type == TYPE.Moon or type == TYPE.Random or type == TYPE.GaussianQuantiles:
            EXPERIMENT.WITH_DRAW_3D = True
            ANALYSIS.WITH_DATASET = True

        EXPERIMENT.PROJ_DIM = proj_dim
        EXPERIMENT.PROJ_VALUES = proj_values

    return config_fn


def main():
    data = dataset()
    run(
        dataset=data,
        net=proj_net(
            net(),
            proj_dims=EXPERIMENT.PROJ_DIM,
            proj_values=EXPERIMENT.PROJ_VALUES,
        ),
    )


if __name__ == "__main__":
    # lab1：深度和神经元和输入维度，如何影响到区域数量的. ReLU激活函数, 不考虑BN.
    lab1_conf = [
        # 数据集Moon
        # Base
        lab1("Linear-[32,32,32]", TYPE.Moon, [32] * 3, nn.Norm1d),
        # 1. 神经元相同, 深度的影响;
        # 我们确保深度的构成是中间层神经元是最大的，依此递增后递减，不会出现夹杂情况
        lab1("Linear-[16,32,32,16]", TYPE.Moon, [16, 32, 32, 16], nn.Norm1d),
        lab1("Linear-[8,8,16,32,16,8,8]", TYPE.Moon, [8, 8, 16, 32, 16, 8, 8], nn.Norm1d),
        # 2. 神经元相同, 宽度的影响(浅层, 中层, 深层);
        lab1("Linear-[64,32,32]", TYPE.Moon, [64, 32, 16], nn.Norm1d),
        lab1("Linear-[16,64,32]", TYPE.Moon, [16, 64, 32], nn.Norm1d),
        lab1("Linear-[32,64,16]", TYPE.Moon, [32, 64, 16], nn.Norm1d),
        lab1("Linear-[16,32,64]", TYPE.Moon, [16, 32, 64], nn.Norm1d),
        # 数据集Random
        # Base
        lab1("Linear-[32,32,32]", TYPE.Random, [32] * 3, nn.Norm1d),
        # 1. 神经元相同, 深度的影响;
        lab1("Linear-[16,32,32,16]", TYPE.Random, [16, 32, 32, 16], nn.Norm1d),
        lab1("Linear-[8,8,16,32,16,8,8]", TYPE.Random, [8, 8, 16, 32, 16, 8, 8], nn.Norm1d),
        # 2. 深度相同, 宽度的影响(浅层, 中层, 深层);
        lab1("Linear-[64,32,32]", TYPE.Random, [64, 32, 16], nn.Norm1d),
        lab1("Linear-[16,64,32]", TYPE.Random, [16, 64, 32], nn.Norm1d),
        lab1("Linear-[32,64,16]", TYPE.Random, [32, 64, 16], nn.Norm1d),
        lab1("Linear-[16,32,64]", TYPE.Random, [16, 32, 64], nn.Norm1d),
        # 3. 深度宽度相同, 输入维度的影响;
        lab1("Linear-[32]x3-feat_3", TYPE.Random, [32] * 3, nn.Norm1d, 3),
        lab1("Linear-[32]x3-feat_4", TYPE.Random, [32] * 3, nn.Norm1d, 4),
    ]

    # lab2：层级中，线性区域中穿越的超平面数量的分析
    lab2_conf = [
        # TODO: 考虑实验设计
        # Moon基础模型分析
        lab2("Linear-[32,32,32]", TYPE.Moon, [32] * 3, nn.Norm1d),
        # 对深层次的模型的分析
        lab2("Linear-[32]x5", TYPE.Moon, [32] * 5, nn.Norm1d),
        lab2("Linear-[32]x10", TYPE.Moon, [32] * 10, nn.Norm1d),
        # Random基础模型分析
        lab2("Linear-[32,32,32]", TYPE.Random, [32] * 3, nn.Norm1d),
        # 对深层次的模型的分析
        lab2("Linear-[32]x5", TYPE.Random, [32] * 5, nn.Norm1d),
        lab2("Linear-[32]x10", TYPE.Random, [32] * 10, nn.Norm1d),
    ]

    # lab3：CNN线性区域的投影面的分析，与splinecam不同；
    lab3_conf = [
        # CIFAI-10, Linear的投影.
        lab3("CIFAR10-Linear", TYPE.CIFAR10, nn.Norm1d, True, 256, (1000, 2000), torch.zeros((3, 32, 32)) + 0.5),
        # CIFAR-10, CNN的投影.
        lab3("CIFAR10-CNN", TYPE.CIFAR10, nn.Norm2d, False, 256, (1000, 2000), torch.zeros((3, 32, 32)) + 0.5),
        # MNIST, CNN的投影.
        lab3("MNIST", TYPE.MNIST, nn.Norm2d, False, 256, (300, 400), torch.zeros((1, 28, 28)) + 0.5),
    ]

    # lab4：BN等神经网络模块对线性区域的影响的实验分析；
    lab4_conf = [
        # 普通神经网络
        lab4("Linear-[32,32,32]-batch", TYPE.Moon, nn.BatchNorm1d, n_layers=[32] * 3),
        lab4("Linear-[32,32,32]-batch", TYPE.Random, nn.BatchNorm1d, n_layers=[32] * 3),
        # CIFAR-10
        lab4("CIFAR10-batch", TYPE.CIFAR10, nn.BatchNorm2d, batch_size=256, proj_dim=(1000, 2000), proj_values=torch.zeros((1, 28, 28)) + 0.5),
    ]

    configs = [
        *lab1_conf,
        *lab2_conf,
        *lab3_conf,
        *lab4_conf,
    ]
    for lab_config in configs:
        lab_config()
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")
