import numpy as np


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def int_ceil(value, epsilon=1e-5) -> int:
    return int(np.ceil(value - epsilon))


def int_floor(value, epsilon=1e-5) -> int:
    return int(np.floor(value + epsilon))
