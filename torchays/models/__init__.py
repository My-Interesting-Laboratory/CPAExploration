from .cifar import CIFARNet
from .mnist import LeNet
from .testnet import TestNetLinear, TestResNet

__all__ = [
    "TestResNet",
    "TestNetLinear",
    "LeNet",
    "CIFARNet",
]
