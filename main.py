import os

import numpy as np
import torch

from dataset import GAUSSIAN_QUANTILES, MNIST, MNIST_TYPE, MOON, RANDOM, CLASSIFICATION, simple_get_data
from experiment import Analysis, Experiment
from torchays import nn
from torchays.models import LeNet, TestResNet, TestNetLinear
from torchays.cpa import ProjectWrapper

GPU_ID = 0
SEED = 5
NAME = "Linear-[32,32,32]"
# ===========================================
TYPE = MOON
# ===========================================
# Net
# Test-Net
N_LAYERS = [32, 32, 32]
# Project, "None"
PROJ_DIM = None
# the values of projection
PROJ_VALUES = torch.zeros((1, 28, 28)) + 0.5
# ===========================================
# Dataset
N_SAMPLES = 1000
DATASET_BIAS = 0
# only GAUSSIAN_QUANTILES
N_CLASSES = 2
# only RANDOM
IN_FEATURES = 2
# is download for mnist
DOWNLOAD = True
# ===========================================
# Training
# is training the network.
IS_TRAIN = True
MAX_EPOCH = 1000
SAVE_EPOCH = [50, 100, 200, 300, 400, 500, 700, 1000]
BATCH_SIZE = 64
LR = 1e-3
# ===========================================
BOUND = (-1, 1)
# the depth of the NN to draw
DEPTH = -1
# the number of the workers
WORKERS = 32
# with best epoch
BEST_EPOCH = False
# ===========================================
# Drawing
# Experiment
IS_EXPERIMENT = True
# is drawing the region picture. Only for 2d input.
IS_DRAW = True
# is drawing the 3d region picture when "IS_DRAW" is True.
IS_DRAW_3D = False
# is handlering the hyperplanes arrangement.
IS_DRAW_HPAS = False
IS_STATISTIC_HPAS = True
# ===========================================
# Analysis
IS_ANALYSIS = False
# draw the dataset distribution
WITH_DATASET = False
# ===========================================
# path
TAG = "batch_norm"

root_dir = os.path.abspath("./")
cache_dir = os.path.join(root_dir, "cache")
if len(TAG) > 0:
    cache_dir = os.path.join(cache_dir, TAG)
save_dir = os.path.join(cache_dir, f"{TYPE}-{N_SAMPLES}-{IN_FEATURES}-{SEED}")


def save_dir(tag: str = "", cache: str = "cache"):
    root_dir = os.path.abspath("./")
    cache_dir = os.path.join(root_dir, cache)
    if tag is not None and len(TAG) > 0:
        cache_dir = os.path.join(cache_dir, tag)
    s_dir = os.path.join(cache_dir, f"{TYPE}-{N_SAMPLES}-{IN_FEATURES}-{SEED}")


def init_fun():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def net(
    type: str = MOON,
    proj_dims: tuple | None = None,
    proj_values: torch.Tensor = None,
):

    def make_net(n_classes: int, training: bool = True):
        if type == MNIST_TYPE:
            return LeNet()
        return TestNetLinear(
            in_features=IN_FEATURES,
            layers=N_LAYERS,
            name=NAME,
            n_classes=n_classes,
            norm_layer=nn.BatchNorm1d,
        )

    if (proj_dims and proj_values) is not None:

        def wrapper(n_classes: int, training: bool = True):
            net = make_net(n_classes, training)
            if training:
                return net
            return ProjectWrapper(
                net,
                proj_dims=proj_dims,
                proj_values=proj_values,
            )

        return wrapper

    return make_net


def dataset(
    save_dir: str,
    type: str = MOON,
    name: str = "dataset.pkl",
):
    def make_dataset():
        if type == MNIST_TYPE:
            mnist = MNIST(root=os.path.join(save_dir, "mnist"), download=DOWNLOAD)
            return mnist, len(mnist.classes)
        return simple_get_data(dataset=type, n_samples=N_SAMPLES, noise=0.2, random_state=5, data_path=os.path.join(save_dir, name), n_classes=N_CLASSES, in_features=IN_FEATURES, bias=DATASET_BIAS)

    return make_dataset


def main():
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    if IS_EXPERIMENT:
        exp = Experiment(
            save_dir=save_dir,
            net=net(type=TYPE, proj_dims=PROJ_DIM, proj_values=PROJ_VALUES),
            dataset=dataset(save_dir, type=TYPE),
            init_fun=init_fun,
            save_epoch=SAVE_EPOCH,
            device=device,
        )
        if IS_TRAIN:
            exp.train(
                max_epoch=MAX_EPOCH,
                batch_size=BATCH_SIZE,
                lr=LR,
            )
        exp.cpas(
            workers=WORKERS,
            best_epoch=BEST_EPOCH,
            bounds=BOUND,
            depth=DEPTH,
            is_draw=IS_DRAW,
            is_draw_3d=IS_DRAW_3D,
            is_draw_hpas=IS_DRAW_HPAS,
            is_statistic_hpas=IS_STATISTIC_HPAS,
        )
        exp()
    if IS_ANALYSIS:
        analysis = Analysis(
            root_dir=save_dir,
            with_dataset=WITH_DATASET,
        )
        analysis()


if __name__ == "__main__":
    main()
