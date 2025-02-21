import os
from typing import Callable

import numpy as np
import torch

from dataset import Dataset
from experiment import Analysis, Experiment
from torchays.cpa import ProjectWrapper, Model
from config import COMMON, GLOBAL, TRAIN, EXPERIMENT, ANALYSIS


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
):
    is_proj = (proj_dims and proj_values) is not None

    def wrapper(n_classes: int, training: bool = True):
        net = new(n_classes)
        if not is_proj or training:
            return net
        return ProjectWrapper(
            net,
            proj_dims=proj_dims,
            proj_values=proj_values,
        )

    return wrapper


def main(
    dataset: Dataset,
    new: Callable[[int], Model],
):
    save_dir = dataset.path
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', COMMON.GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    if EXPERIMENT.EXPERIMENT:
        exp = Experiment(
            save_dir=save_dir,
            net=proj_net(
                new=new,
                proj_dims=EXPERIMENT.PROJ_DIM,
                proj_values=EXPERIMENT.PROJ_VALUES,
            ),
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
            )
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
        exp()
    if ANALYSIS.WITH_ANALYSIS:
        analysis = Analysis(
            root_dir=save_dir,
            with_dataset=ANALYSIS.WITH_DATASET,
        )
        analysis()


if __name__ == "__main__":
    main(
        GLOBAL.dataset(),
        GLOBAL.net(),
    )
