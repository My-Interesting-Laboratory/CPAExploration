import os
from typing import Tuple

from PIL import Image
from torch.utils import data
from torchvision.datasets import cifar

from .dataset import Dataset
from .utils import transform as _transform


class CIFAR10(cifar.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        if self.transform is None:
            self.transform = _transform
        self.input_size = (3, 32, 32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Cifar10(Dataset):
    def __init__(self, root: str, download: bool = True):
        super().__init__("cifar10", root)
        self.download = download

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        cifar10 = CIFAR10(root=self.path, download=self.download)
        return cifar10, len(cifar10.classes)

    def make_path(self):
        return os.path.join(self.root, self.type)
