import os
from typing import List, Tuple

import torch

from torchays import nn


class TYPE:
    Moon = "Moon"
    GaussianQuantiles = "GaussianQuantiles"
    Random = "Random"
    Classification = "Classification"
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"


class COMMON:
    GPU_ID: int = 7
    # random seed
    SEED: int = 5


class PATH:
    TAG: str = "cpas-norm"
    DIR: str = os.path.join(os.path.abspath("./"), "cache")


class TOY:
    # All toy datasets
    IN_FEATURES: int = 2
    N_SAMPLES: int = 100
    N_CLASSES: int = 2
    BIAS: float = 0
    # Moon
    NOISE: float = 0.2


class MNIST:
    DOWNLOAD: bool = True


class CIFAR10:
    DOWNLOAD: bool = True
    LINEAR: bool = False
    NORM_LAYER = nn.BatchNorm2d


class TESTNET:
    N_LAYERS = [64] * 3
    NORM_LAYER = nn.BatchNorm1d


class GLOBAL:
    NAME: str = "Linear-[64]x3-batch-no_l2"
    # ["Moon", "GaussianQuantiles", "Random", "Classification", "MNIST"]
    TYPE: str = "Random"


class TRAIN:
    TRAIN: bool = True
    MAX_EPOCH: int = 10000
    SAVE_EPOCH: List[int] = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 30000]
    BATCH_SIZE: int = 64
    LR: float = 1e-3


class EXPERIMENT:
    # cpas
    CPAS: bool = True
    # point
    POINT: bool = False
    # Project, "None"
    PROJ_DIM: Tuple[int] | None = None
    # The values of projection including "PROJ_DIM" which is useless.
    PROJ_VALUES: torch.Tensor = torch.zeros((1, 28, 28)) + 0.5
    # Bound for find cpa
    BOUND: Tuple[int] = (-1, 1)
    # The depth of the NN to draw
    DEPTH: int = -1
    # The number of the workers
    WORKERS: int = 1
    # With best epoch
    WITH_BEST: bool = False
    # Drawing
    # is drawing the region picture. Only for 2d input.
    WITH_DRAW: bool = True
    # is drawing the 3d region picture when "IS_DRAW" is True.
    WITH_DRAW_3D: bool = False
    # is handlering the hyperplanes arrangement.
    WITH_DRAW_HPAS: bool = False
    WITH_STATISTIC_HPAS: bool = False


class ANALYSIS:
    # Analysis
    WITH_ANALYSIS: bool = False
    # draw the dataset distribution
    WITH_DATASET: bool = False
