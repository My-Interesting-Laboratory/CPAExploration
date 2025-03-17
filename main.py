from config import ANALYSIS, EXPERIMENT, GLOBAL, PATH, TESTNET, TOY, TRAIN
from run import dataset, net, proj_net, run
from torchays import nn


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

    TRAIN.TRAIN = False

    TESTNET.N_LAYERS = [64] * 5

    EXPERIMENT.CPAS = False
    EXPERIMENT.POINT = True
    EXPERIMENT.WORKERS = 64


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
    configs = [
        ("Linear-[64]x5-norm", "Random", nn.Norm1d),
        ("Linear-[64]x5-batch", "Random", nn.BatchNorm1d),
        ("Linear-[64]x5-norm", "Moon", nn.Norm1d),
        ("Linear-[64]x5-batch", "Moon", nn.BatchNorm1d),
    ]
    for cfg in configs:
        config()
        set_config(*cfg)
        print(f"========= Now: {GLOBAL.NAME} =========")
        main()
        print(f"========= End: {GLOBAL.NAME} =========\n")
