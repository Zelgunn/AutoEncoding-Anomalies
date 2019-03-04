from abc import ABC, abstractmethod
import numpy as np


class DataPreprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        raise NotImplementedError


def random_range_value(range_or_min_max, center=1.0, size=None):
    if hasattr(range_or_min_max, "__getitem__"):
        min_value, max_value = random_range_value
    else:
        min_value = center - range_or_min_max * 0.5
        max_value = center + range_or_min_max * 0.5

    return np.random.uniform(size=size) * (max_value - min_value) + min_value
