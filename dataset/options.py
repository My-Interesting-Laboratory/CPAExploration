__all__ = [
    "Moon",
    "GaussianQuantiles",
    "Random",
    "Classification",
    "Mnist",
]


class Options:
    type: str

    def __str__(self):
        return self.type


class Moon(Options):
    pass


class GaussianQuantiles(Options):
    pass


class Random(Options):
    pass


class Classification(Options):
    pass


class Mnist(Options):
    pass
