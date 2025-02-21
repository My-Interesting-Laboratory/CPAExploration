from typing import Tuple

from torch.utils import data


class Dataset:
    type: str
    root: str
    _path: str = None

    def __init__(self, type: str, root: str):
        self.type = type
        self.root = root

    def __str__(self):
        return self.type

    @property
    def path(self) -> str:
        if self._path is None:
            self._path = self.make_path()
        return self._path

    def make_path(self) -> str:
        raise NotImplementedError()

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        raise NotImplementedError()
