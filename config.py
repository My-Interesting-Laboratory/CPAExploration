import os
from typing import List, Tuple

import torch

from dataset import Classification, GaussianQuantiles, Mnist, Moon, Random
from torchays.cpa import Model
from torchays.models import LeNet, TestNetLinear
from torchays import nn


class COMMON:
    GPU_ID: int = 7
    # random seed
    SEED: int = 5


class PATH:
    TAG: str = "cpas-batch"
    DIR: str = os.path.join(os.path.abspath("./"), "cache")
    ROOT: str = os.path.join(DIR, TAG) if TAG is not None and len(TAG) > 0 else DIR


class TOY:
    # All toy datasets
    IN_FEATURES: int = 2
    N_SAMPLES: int = 100
    N_CLASS: int = 2
    BIAS: float = 0
    # Moon
    NOISE: float = 0.2


class MNIST:
    DOWNLOAD: bool = True


class TESTNET:
    NAME: str = "Linear-[64]x3"
    N_LAYERS = [64] * 3
    NORM_LAYER = nn.BatchNorm1d


class GLOBAL:
    # ["Moon", "GaussianQuantiles", "Random", "Classification", "MNIST"]
    TYPE: str = "Random"

    @classmethod
    def dataset(clz):
        dataset = None
        if clz.TYPE == "Moon":
            dataset = Moon(root=PATH.ROOT, n_samples=TOY.N_SAMPLES, noise=TOY.NOISE, random_state=COMMON.SEED, bias=TOY.BIAS)
        if clz.TYPE == "GaussianQuantiles":
            dataset = GaussianQuantiles(root=PATH.ROOT, n_samples=TOY.N_SAMPLES, n_classes=TOY.N_CLASS, bias=TOY.BIAS, random_state=COMMON.SEED)
        if clz.TYPE == "Random":
            dataset = Random(root=PATH.ROOT, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, bias=TOY.BIAS)
        if clz.TYPE == "Classification":
            dataset = Classification(root=PATH.ROOT, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, n_classes=TOY.N_CLASS, bias=TOY.BIAS, random_state=COMMON.SEED)
        if clz.TYPE == "MNIST":
            dataset = Mnist(PATH.ROOT, MNIST.DOWNLOAD)
        return dataset

    @classmethod
    def net(clz):

        def wrapper(n_classes: int) -> Model:
            if clz.TYPE == "MNIST":
                # LeNet
                model = LeNet()
            else:
                # Toy
                model = TestNetLinear(
                    in_features=TOY.IN_FEATURES,
                    layers=TESTNET.N_LAYERS,
                    name=TESTNET.NAME,
                    n_classes=n_classes,
                    norm_layer=TESTNET.NORM_LAYER,
                )
            return model

        return wrapper


class TRAIN:
    TRAIN: bool = True
    MAX_EPOCH: int = 5000
    SAVE_EPOCH: List[int] = [100, 500, 1000, 2500, 5000]
    BATCH_SIZE: int = 64
    LR: float = 1e-3


class EXPERIMENT:
    # Experiment
    EXPERIMENT: bool = True
    # Project, "None"
    PROJ_DIM: Tuple[int] | None = None
    # The values of projection
    PROJ_VALUES: torch.Tensor = torch.zeros((1, 28, 28)) + 0.5
    # Bound for find cpa
    BOUND: Tuple[int] = (-1, 1)
    # The depth of the NN to draw
    DEPTH: int = -1
    # The number of the workers
    WORKERS: int = 64
    # With best epoch
    WITH_BEST: bool = True
    # Drawing
    # is drawing the region picture. Only for 2d input.
    WITH_DRAW: bool = True
    # is drawing the 3d region picture when "IS_DRAW" is True.
    WITH_DRAW_3D: bool = False
    # is handlering the hyperplanes arrangement.
    WITH_DRAW_HPAS: bool = False
    WITH_STATISTIC_HPAS: bool = True


class ANALYSIS:
    # Analysis
    WITH_ANALYSIS: bool = False
    # draw the dataset distribution
    WITH_DATASET: bool = False
