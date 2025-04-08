import os
from typing import Callable

import numpy as np
import torch

from config import ANALYSIS, COMMON, EXPERIMENT, GLOBAL, MNIST, PATH, TESTNET, TOY, TRAIN
from dataset import Classification, Dataset, GaussianQuantiles, Mnist, Moon, Random
from experiment import Analysis, Experiment, TrainHandler
from torchays import nn
from torchays.cpa import Model, ProjectWrapper
from torchays.models import LeNet, TestNetLinear


def init_fun(seed: int = 0):
    def init():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    return init


def proj_net(
    new: Callable[[int], Model],
    proj_dims: tuple | None = None,
    proj_values: torch.Tensor = None,
    handler: Callable[[str, torch.Tensor, str], None] = None,
):
    is_proj = (proj_dims and proj_values) is not None

    def wrapper(n_classes: int, training: bool = True):
        net = new(n_classes)
        if handler is not None and training:
            net.handler = handler
        if not is_proj or training:
            return net
        return ProjectWrapper(
            net,
            proj_dims=proj_dims,
            proj_values=proj_values,
        )

    return wrapper


def dataset():
    root: str = os.path.join(PATH.DIR, PATH.TAG) if PATH.TAG is not None and len(PATH.TAG) > 0 else PATH.DIR
    dataset = None
    if GLOBAL.TYPE == "Moon":
        dataset = Moon(root=root, n_samples=TOY.N_SAMPLES, noise=TOY.NOISE, random_state=COMMON.SEED, bias=TOY.BIAS)
    if GLOBAL.TYPE == "GaussianQuantiles":
        dataset = GaussianQuantiles(root=root, n_samples=TOY.N_SAMPLES, n_classes=TOY.N_CLASSES, bias=TOY.BIAS, random_state=COMMON.SEED)
    if GLOBAL.TYPE == "Random":
        dataset = Random(root=root, n_classes=TOY.N_CLASSES, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, random_state=COMMON.SEED, bias=TOY.BIAS)
    if GLOBAL.TYPE == "Classification":
        dataset = Classification(root=root, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, n_classes=TOY.N_CLASSES, bias=TOY.BIAS, random_state=COMMON.SEED)
    if GLOBAL.TYPE == "MNIST":
        dataset = Mnist(root, MNIST.DOWNLOAD)
    return dataset


def net():

    def wrapper(n_classes: int) -> Model:
        if GLOBAL.TYPE == "MNIST":
            # LeNet
            model = LeNet()
        else:
            # Toy
            model = TestNetLinear(
                in_features=TOY.IN_FEATURES,
                layers=TESTNET.N_LAYERS,
                name=GLOBAL.NAME,
                n_classes=n_classes,
                norm_layer=TESTNET.NORM_LAYER,
            )
        return model

    return wrapper


def print_cfg():
    print("------------------")
    print("Configuration:")
    print(f"Name: {GLOBAL.NAME}")
    print(f"Random Seed: {COMMON.SEED}")
    print(f"Dataset: {GLOBAL.TYPE}")
    if GLOBAL.TYPE == "MNIST":
        print(f"Net: LeNet")
    else:
        print(f"|   n_samples: {TOY.N_SAMPLES}")
        print(f"|   n_class: {TOY.N_CLASSES}")
        print(f"|   in_feature: {TOY.IN_FEATURES}")
        print(f"Net: TestNetLinear")
        print(f"|   Layers: {TESTNET.N_LAYERS}")
        print(f"|   Norm Layer: {TESTNET.NORM_LAYER.__name__}")
    print(f"Action:")
    print(f"|   Train: {TRAIN.TRAIN}")
    print(f"|   CPAS: {EXPERIMENT.CPAS}")
    print(f"|   Point: {EXPERIMENT.POINT}")
    print("------------------")


def run(
    *,
    dataset: Dataset,
    net: Callable[[int, bool], Model],
    init_fun: Callable[[int], None] = init_fun,
    train_handler: TrainHandler = None,
):
    print_cfg()
    save_dir = dataset.path
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', COMMON.GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    exp = Experiment(
        save_dir=save_dir,
        net=net,
        dataset=dataset.make_dataset,
        init_fun=init_fun(COMMON.SEED),
        save_epoch=TRAIN.SAVE_EPOCH,
        device=device,
    )
    if TRAIN.TRAIN:
        exp.train(
            max_epoch=TRAIN.MAX_EPOCH,
            batch_size=TRAIN.BATCH_SIZE,
            lr=TRAIN.LR,
            train_handler=train_handler,
        )
    if EXPERIMENT.CPAS:
        exp.cpas(
            workers=EXPERIMENT.WORKERS,
            best_epoch=EXPERIMENT.WITH_BEST,
            bounds=EXPERIMENT.BOUND,
            depth=EXPERIMENT.DEPTH,
            is_draw=EXPERIMENT.WITH_DRAW,
            is_draw_3d=EXPERIMENT.WITH_DRAW_3D,
            is_draw_hpas=EXPERIMENT.WITH_DRAW_HPAS,
            is_statistic_hpas=EXPERIMENT.WITH_STATISTIC_HPAS,
        )
    if EXPERIMENT.POINT:
        exp.points(
            best_epoch=EXPERIMENT.WITH_BEST,
            bounds=EXPERIMENT.BOUND,
            depth=EXPERIMENT.DEPTH,
        )
    # run
    exp.run()

    # Analysis
    analysis = Analysis(
        root_dir=save_dir,
        with_analysis=ANALYSIS.WITH_ANALYSIS,
        with_dataset=ANALYSIS.WITH_DATASET,
    )
    analysis()
