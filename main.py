from typing import List, Tuple
import torch
from config import ANALYSIS, EXPERIMENT, GLOBAL, PATH, TESTNET, TOY, TRAIN, TYPE
from run import dataset, net, proj_net, run
from torchays import nn


def set_config(
    tag: str,
    name: str,
    n_layers: List[int],
    type: str,
    norm_layer,
    with_draw: bool = True,
    in_features: int = 2,
    n_samples: int = 100,
    batch_size: int = 64,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    PATH.TAG = tag

    GLOBAL.NAME = name
    GLOBAL.TYPE = type

    TOY.IN_FEATURES = in_features
    TOY.N_SAMPLES = n_samples

    TRAIN.BATCH_SIZE = batch_size

    TESTNET.N_LAYERS = n_layers
    TESTNET.NORM_LAYER = norm_layer

    EXPERIMENT.WITH_DRAW = with_draw
    EXPERIMENT.PROJ_DIM = proj_dim
    EXPERIMENT.PROJ_VALUES = proj_values


def config():
    PATH.TAG = "CPAExploration"

    TOY.N_SAMPLES = 100
    TOY.N_CLASSES = 2

    TRAIN.TRAIN = True
    TRAIN.MAX_EPOCH = 5000
    TRAIN.SAVE_EPOCH = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 30000]
    TRAIN.LR = 1e-3

    EXPERIMENT.CPAS = False
    EXPERIMENT.POINT = False
    EXPERIMENT.WITH_DRAW_3D = False
    EXPERIMENT.WITH_DRAW_HPAS = False
    EXPERIMENT.WITH_STATISTIC_HPAS = False
    EXPERIMENT.WORKERS = 64
    EXPERIMENT.BOUND = (-1, 1)
    EXPERIMENT.WORKERS = 64

    ANALYSIS.WITH_DATASET = False


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


def lab1(
    name: str,
    n_layers: List[int],
    type: str,
    norm_layer,
    in_features: int = 2,
    n_samples: int = 200,
):
    with_draw = True
    if in_features > 2:
        with_draw = False
    return ("CPAExploration-lab1", name, n_layers, type, norm_layer), {"with_draw": with_draw, "in_features": in_features, "n_samples": n_samples}


def lab2(
    name: str,
    n_layers: List[int],
    type: str,
    norm_layer,
    in_features: int = 2,
):
    return ("CPAExploration-lab2", name, n_layers, type, norm_layer), {"in_features": in_features}


def lab3(
    name: str,
    n_layers: List[int],
    type: str,
    norm_layer,
    batch_size: int = 64,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    return ("CPAExploration-lab3", name, n_layers, type, norm_layer), {"proj_dim": proj_dim, "proj_values": proj_values, "batch_size": batch_size}


def lab4(
    name: str,
    n_layers: List[int],
    type: str,
    norm_layer,
    batch_size: int = 64,
    proj_dim: Tuple[int] = None,
    proj_values: torch.Tensor = None,
):
    return ("CPAExploration-lab4", name, n_layers, type, norm_layer), {"proj_dim": proj_dim, "proj_values": proj_values, "batch_size": batch_size}


if __name__ == "__main__":
    # lab1：深度和神经元和输入维度，如何影响到区域数量的. ReLU激活函数, 不考虑BN.
    lab1_conf = [
        # 数据集Moon
        # Base
        lab1("Linear-[32,32,32]", [32] * 3, TYPE.Moon, nn.Norm1d),
        # 1. 神经元相同, 深度的影响;
        # 我们确保深度的构成是中间层神经元是最大的，依此递增后递减，不会出现夹杂情况
        lab1("Linear-[16,32,32,16]", [16, 32, 32, 16], TYPE.Moon, nn.Norm1d),
        lab1("Linear-[8,8,16,32,16,8,8]", [8, 8, 16, 32, 16, 8, 8], TYPE.Moon, nn.Norm1d),
        # 2. 神经元相同, 宽度的影响(浅层, 中层, 深层);
        lab1("Linear-[64,32,32]", [64, 32, 16], TYPE.Moon, nn.Norm1d),
        lab1("Linear-[16,64,32]", [16, 64, 32], TYPE.Moon, nn.Norm1d),
        lab1("Linear-[32,64,16]", [32, 64, 16], TYPE.Moon, nn.Norm1d),
        lab1("Linear-[16,32,64]", [16, 32, 64], TYPE.Moon, nn.Norm1d),
        # 数据集Random
        # Base
        lab1("Linear-[32,32,32]", [32] * 3, TYPE.Random, nn.Norm1d),
        # 1. 神经元相同, 深度的影响;
        lab1("Linear-[16,32,32,16]", [16, 32, 32, 16], TYPE.Random, nn.Norm1d),
        lab1("Linear-[8,8,16,32,16,8,8]", [8, 8, 16, 32, 16, 8, 8], TYPE.Random, nn.Norm1d),
        # 2. 深度相同, 宽度的影响(浅层, 中层, 深层);
        lab1("Linear-[64,32,32]", [64, 32, 16], TYPE.Random, nn.Norm1d),
        lab1("Linear-[16,64,32]", [16, 64, 32], TYPE.Random, nn.Norm1d),
        lab1("Linear-[32,64,16]", [32, 64, 16], TYPE.Random, nn.Norm1d),
        lab1("Linear-[16,32,64]", [16, 32, 64], TYPE.Random, nn.Norm1d),
        # 3. 深度宽度相同, 输入维度的影响;
        lab1("Linear-[32]x3-feat_3", [32] * 3, TYPE.Random, nn.Norm1d, 3),
        lab1("Linear-[32]x3-feat_4", [32] * 3, TYPE.Random, nn.Norm1d, 4),
    ]

    # lab2：层级中，线性区域中穿越的超平面数量的分析
    lab2_conf = [
        # TODO: 考虑实验设计
        lab2("Linear-[32]x3", [32] * 3, TYPE.Random, nn.Norm1d),
    ]

    # lab3：CNN线性区域的投影面的分析，与splinecam不同；
    # lab3：随机抽取jige
    lab3_conf = [
        # CIFAR-10, BN, 不同维度的投影.
        # lab3("CIFAR10", None, TYPE.CIFAR10, nn.Norm2d, 256, (1000, 2000), torch.zeros((3, 32, 32)) + 0.5),
        # MNIST, BN, 不同维度的投影.
        lab3("MNIST", None, TYPE.MNIST, nn.Norm2d, 256, (300, 400), torch.zeros((1, 28, 28)) + 0.5),
    ]

    # lab4：BN等神经网络模块对线性区域的影响的实验分析；
    lab4_conf = [
        # 普通神经网络
        lab4("Linear-[32,32,32]x3-batch", TYPE.Moon, [32] * 3, nn.BatchNorm1d),
        lab4("Linear-[32,32,32]x3-batch", TYPE.Random, [32] * 3, nn.BatchNorm1d),
        # CIFAR-10
        lab4("CIFAR10-batch", None, TYPE.CIFAR10, nn.BatchNorm2d, (100, 101), torch.zeros((1, 28, 28)) + 0.5),
        # MNIST
        lab4("MNIST-batch", None, TYPE.MNIST, nn.BatchNorm2d, (100, 101), torch.zeros((1, 28, 28)) + 0.5),
    ]

    configs = [
        *lab1_conf,
        # *lab2_conf,
        # *lab3_conf,
        # *lab4_conf,
    ]
    for args, kwargs in configs:
        config()
        set_config(*args, **kwargs)
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")
