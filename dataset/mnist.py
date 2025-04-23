import os
from typing import Any, Tuple

from PIL import Image
from torch.utils import data
from torchvision.datasets import mnist

from torchvision.transforms import Compose, ToTensor, Normalize

from .dataset import Dataset


def _transform(img: Image.Image):
    trans = Compose(
        [
            ToTensor(),
            Normalize((0.1307), (0.3081)),
        ]
    )
    return trans(img)


class MNIST(mnist.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.input_size = (1, 28, 28)


class Mnist(Dataset):
    def __init__(self, root: str, download: bool = True):
        super().__init__("mnist", root)
        self.download = download

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        mnist = MNIST(root=self.path, transform=_transform, download=self.download)
        return mnist, len(mnist.classes)

    def make_path(self):
        return os.path.join(self.root, self.type)
