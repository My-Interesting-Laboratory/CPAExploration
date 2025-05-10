import random
from typing import Iterable, List, Tuple

import torch

from config import ANALYSIS, CIFAR10, COMMON, EXPERIMENT, GLOBAL, PATH, TESTNET, TOY, TRAIN, TYPE
from run import dataset, net, proj_net, run
from torchays import nn


def config():
    COMMON.GPU_ID = 7

    TRAIN.TRAIN = True
    TRAIN.MAX_EPOCH = 5000
    TRAIN.LR = 1e-4
    TRAIN.SAVE_EPOCH = [1, 10, 50, 100, 200, 300, 500, 1000, 2000, 5000, 7500, 10000, 20000, 30000]

    EXPERIMENT.CPAS = True
    EXPERIMENT.WORKERS = 64
    EXPERIMENT.BOUND = (-1, 1)

    EXPERIMENT.WITH_DRAW = True
    EXPERIMENT.WITH_DRAW_3D = False
    EXPERIMENT.WITH_DRAW_HPAS = False
    EXPERIMENT.WITH_STATISTIC_HPAS = False

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
    type: str,
    n_layers: List[int],
    norm_layer=nn.Norm1d,
    in_features: int = 2,
    n_classes: int = 2,
):
    name = f"Linear-{n_layers}".replace(" ", "")
    if in_features > 2:
        name += f"-feat_{in_features}"
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

        EXPERIMENT.WITH_DRAW_3D = False if in_features > 2 else True
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
        EXPERIMENT.WITH_DRAW_HPAS = False
        EXPERIMENT.WITH_STATISTIC_HPAS = True

        ANALYSIS.WITH_DATASET = True

    return config_fn


def lab3(
    name: str,
    type: str,
    norm_layer,
    linear: bool,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    args = (name, type)
    kwargs = {"batch_size": 256}

    def config_fn():
        PATH.TAG = "CPAExploration-lab3"
        set_config(*args, **kwargs)
        TRAIN.MAX_EPOCH = 500

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


def random_proj(input_size: torch.Size | Iterable[int], dims: int = 2, bound=(-1, 1)):
    if isinstance(input_size, Iterable):
        input_size = torch.Size(input_size)
    kwargs = {}
    numel = input_size.numel()
    # proj_dim
    proj_dim = []
    for _ in range(dims):
        while True:
            dim = random.randint(0, numel)
            if dim not in proj_dim:
                proj_dim.append(dim)
                break
    kwargs["proj_dim"] = tuple(proj_dim)
    # proj_values
    upper, lower = max(bound), min(bound)
    kwargs["proj_values"] = torch.rand(*input_size) * (upper - lower) + lower
    return kwargs


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
        # ====================================================================
        # Moon
        # 1. 单层MLP情况下, 随着宽度增加, 区域数量的增长关系; (宽度于区域数量的关系)
        *[lab1(TYPE.Moon, [16 * (2**i)]) for i in range(0, 6)],
        # 2. 多层MLP情况下, 每一层神经元数量固定, 如何影响到区域数量;
        *[lab1(TYPE.Moon, [16] * i) for i in range(2, 10)],
        # 3. 多层MLP情况下, 随着某一层的宽度变动, 区域数量的增加关系;
        *[lab1(TYPE.Moon, [16, 16 * (2**i)]) for i in range(0, 5)],
        *[lab1(TYPE.Moon, [16 * (2**i), 16]) for i in range(0, 5)],
        # ====================================================================
        # Random-feat_4
        # 1. 单层MLP情况下, 随着宽度增加, 区域数量的增长关系;
        *[lab1(TYPE.Random, [16 * (2**i)]) for i in range(0, 6)],
        # 2. 多层MLP情况下, 每一层神经元数量固定, 如何影响到区域数量;
        *[lab1(TYPE.Random, [16] * i) for i in range(2, 10)],
        # 3. 多层MLP情况下, 随着某一层的宽度变动, 区域数量的增加关系;
        *[lab1(TYPE.Random, [16, 16 * (2**i)]) for i in range(0, 5)],
        *[lab1(TYPE.Random, [16 * (2**i), 16]) for i in range(0, 5)],
        # ====================================================================
        # Random-feat_3
        # 1. 单层MLP情况下, 随着宽度增加, 区域数量的增长关系;
        *[lab1(TYPE.Random, [16 * (2**i)], in_features=3) for i in range(0, 2)],
        # 2. 多层MLP情况下, 每一层神经元数量固定, 如何影响到区域数量;
        *[lab1(TYPE.Random, [16] * i, in_features=3) for i in range(2, 4)],
        # 3. 多层MLP情况下, 随着某一层的宽度变动, 区域数量的增加关系;
        *[lab1(TYPE.Random, [16, 16 * (2**i)], in_features=3) for i in range(0, 1)],
        *[lab1(TYPE.Random, [16 * (2**i), 16], in_features=3) for i in range(0, 1)],
        # ====================================================================
        # Random-feat_4
        # 1. 单层MLP情况下, 随着宽度增加, 区域数量的增长关系;
        *[lab1(TYPE.Random, [16 * (2**i)], in_features=4) for i in range(0, 2)],
        # 2. 多层MLP情况下, 每一层神经元数量固定, 如何影响到区域数量;
        *[lab1(TYPE.Random, [16] * i, in_features=4) for i in range(2, 4)],
        # 3. 多层MLP情况下, 随着某一层的宽度变动, 区域数量的增加关系;
        *[lab1(TYPE.Random, [16, 16 * (2**i)], in_features=4) for i in range(1, 4)],
        *[lab1(TYPE.Random, [16 * (2**i), 16], in_features=4) for i in range(1, 4)],
    ]

    # lab2：层级中, 线性区域中穿越的超平面数量的分析;(再议)
    lab2_conf = [
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
        # 64
        # Moon基础模型分析
        lab2("Linear-[64,64,64]", TYPE.Moon, [64] * 3, nn.Norm1d),
        # 对深层次的模型的分析
        lab2("Linear-[64]x5", TYPE.Moon, [64] * 5, nn.Norm1d),
        lab2("Linear-[64]x10", TYPE.Moon, [64] * 10, nn.Norm1d),
        # Random基础模型分析
        lab2("Linear-[64,64,64]", TYPE.Random, [64] * 3, nn.Norm1d),
        # 对深层次的模型的分析
        lab2("Linear-[64]x5", TYPE.Random, [64] * 5, nn.Norm1d),
        lab2("Linear-[64]x10", TYPE.Random, [64] * 10, nn.Norm1d),
    ]

    # lab3：CNN线性区域的投影面的分析, 与splinecam不同;
    lab3_conf = [
        # CIFAI-10, Linear的投影.
        *[lab3("CIFAR10-Linear", TYPE.CIFAR10, nn.Norm1d, linear=True, **random_proj((3, 32, 32))) for _ in range(10)],
        # CIFAR-10, CNN的投影.
        *[lab3("CIFAR10-CNN", TYPE.CIFAR10, nn.Norm2d, linear=False, **random_proj((3, 32, 32))) for _ in range(10)],
    ]

    # lab4：BN等神经网络模块对线性区域的影响的实验分析；
    lab4_conf = [
        # 普通神经网络
        lab4("Linear-[32,32,32]-batch", TYPE.Moon, nn.BatchNorm1d, n_layers=[32] * 3),
        lab4("Linear-[32,32,32]-batch", TYPE.Moon, nn.BatchNorm1d, n_layers=[32] * 3),
        lab4("Linear-[64,64,64]-batch", TYPE.Random, nn.BatchNorm1d, n_layers=[64] * 3),
        lab4("Linear-[64,64,64]-batch", TYPE.Random, nn.BatchNorm1d, n_layers=[64] * 3),
        # CIFAR-10
        lab4("CIFAR10-batch", TYPE.CIFAR10, nn.BatchNorm2d, batch_size=256, proj_dim=(1000, 2000), proj_values=torch.zeros((3, 32, 32)) + 0.5),
    ]

    configs = [
        *lab1_conf,
        # *lab2_conf,
        # *lab3_conf,
        # *lab4_conf,
    ]
    for lab_config in configs:
        lab_config()
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")
