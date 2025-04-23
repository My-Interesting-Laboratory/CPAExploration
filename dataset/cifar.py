import os
from typing import Tuple

from PIL import Image
from torch.utils import data
from torchvision.datasets import cifar
from torchvision.transforms import Compose, ToTensor, Normalize

from .dataset import Dataset


def _transform(img: Image.Image):
    trans = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return trans(img)


class CIFAR10(cifar.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.input_size = (3, 32, 32)


class Cifar10(Dataset):
    def __init__(self, root: str, download: bool = True):
        super().__init__("cifar10", root)
        self.download = download

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        cifar10 = CIFAR10(root=self.path, transform=_transform, download=self.download)
        return cifar10, len(cifar10.classes)

    def make_path(self):
        return os.path.join(self.root, self.type)
