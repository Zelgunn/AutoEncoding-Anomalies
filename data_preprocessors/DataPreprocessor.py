from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, inputs, outputs):
        raise NotImplementedError
