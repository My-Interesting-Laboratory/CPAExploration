import os
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification, make_gaussian_quantiles, make_moons
from torch.utils import data

from .dataset import Dataset


class ToyDataset(data.Dataset):
    def __init__(self, data: np.ndarray, classes: np.ndarray) -> None:
        super().__init__()
        self.data, self.classes = data, classes
        self.input_size = self.data.shape[1:]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = torch.from_numpy(self.data[index]), self.classes[index]
        return x, target

    def __len__(self):
        return self.data.shape[0]


def _norm(data: np.ndarray) -> np.ndarray:
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - data.mean(0, keepdims=True)) / ((data.std(0, keepdims=True) + 1e-16))
    data /= np.max(np.abs(data))
    return data


class Toy(Dataset):
    n_samples: int
    _name: str = None
    bias: str

    def __init__(
        self,
        type: str,
        path: str,
        n_samples: int = 1000,
        bias: float = 0,
    ):
        super().__init__(type, path)
        self.n_samples = n_samples
        self.bias = bias
        self.seed: str = np.random.get_state()[1][0]

    def make_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def _make_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        data, classes, n_classes = self.make_data()
        return data + self.bias, classes, n_classes

    def save_data(self, path: str):
        data, classes, n_classes = self._make_data()
        torch.save(
            {
                "data": data,
                "classes": classes,
                "n_classes": n_classes,
            },
            path,
        )
        return ToyDataset(data, classes), n_classes

    def from_path(self, path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"cannot find the dataset file [{self.path}]")
        data_dict: Dict = torch.load(path, weights_only=False)
        data: np.ndarray = data_dict.get("data")
        classes: np.ndarray = data_dict.get("classes")
        n_classes: int = data_dict.get("n_classes")
        return ToyDataset(data, classes), n_classes

    def make_path(self):
        return os.path.join(self.root, self.name())

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        path = os.path.join(self.path, "dataset.pkl")
        if os.path.exists(path):
            return self.from_path(path)
        return self.save_data(path)

    def __str__(self):
        return self.name


class Moon(Toy):
    def __init__(
        self,
        root: str,
        *,
        n_samples: int = 1000,
        noise: float = None,
        random_state: int = None,
        bias: int = 0,
        norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
    ):
        super().__init__("moon", root, n_samples, bias)
        self.noise = noise
        self.random_state = random_state
        self.norm_func = norm_func

    def make_data(self):
        data, classes = make_moons(self.n_samples, noise=self.noise, random_state=self.random_state)
        if self.norm_func is not None:
            data = self.norm_func(data)
        return data, classes, 2

    def name(self):
        return f"{self.type}-{self.n_samples}-{self.seed}"


class GaussianQuantiles(Toy):
    def __init__(
        self,
        root: str,
        *,
        n_samples: int = 1000,
        mean: np.ndarray | None = None,
        cov: float = 1,
        n_features: int = 2,
        n_classes: int = 3,
        shuffle: bool = True,
        random_state: int | None = None,
        bias: int = 0,
        norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
    ):
        super().__init__("gaussian_quantiles", root, n_samples, bias)
        self.mean = mean
        self.cov = cov
        self.n_features = n_features
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.bias = bias
        self.norm_func = norm_func

    def make_data(self):
        data, classes = make_gaussian_quantiles(
            mean=self.mean,
            cov=self.cov,
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        if self.norm_func is not None:
            data = self.norm_func(data)
        return data, classes, self.n_classes

    def name(self):
        return f"{self.type}-{self.n_samples}-{self.n_features}-{self.seed}"


class Random(Toy):
    def __init__(
        self,
        root: str,
        *,
        n_samples: int = 1000,
        in_features: int = 2,
        bias: float = 0,
    ):
        super().__init__("random", root, n_samples, bias)
        self.in_features = in_features

    def make_data(self):
        data = np.random.uniform(-1, 1, (self.n_samples, self.in_features))
        classes = np.sign(np.random.uniform(-1, 1, [self.n_samples]))
        classes = np.where(classes > 0, 1, 0)
        return data, classes, 2

    def name(self):
        return f"{self.type}-{self.n_samples}-{self.in_features}-{self.seed}"


class Classification(Toy):
    def __init__(
        self,
        root: str,
        *,
        n_samples: int | Tuple[int, int] = 1000,
        in_features: int = 2,
        n_classes: int = 3,
        bias: int = 0,
        random_state: int | None = None,
        norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
    ):
        super().__init__("classification", root, n_samples, bias)
        self.in_features = in_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.norm_func = norm_func

    def make_data(self):
        data, classes = make_classification(
            self.n_samples,
            n_features=self.in_features,
            n_informative=self.in_features,
            n_clusters_per_class=1,
            n_redundant=0,
            n_classes=self.n_classes,
            class_sep=10,
            random_state=self.random_state,
            hypercube=True,
        )
        if self.norm_func is not None:
            data = self.norm_func(data)
        return data, classes, self.n_classes

    def name(self):
        return f"{self.type}-{self.n_samples}-{self.in_features}-{self.seed}"
