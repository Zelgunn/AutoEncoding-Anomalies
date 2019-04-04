import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any


class ModalityBuilder(ABC):
    def __init__(self, modalities: Dict[str, Dict[str, Any]]):
        for modality in modalities:
            assert "shard_size" in modalities[modality], "`shard_size` not in modality `{}`".format(modality)

        self.modalities = modalities

    @classmethod
    @abstractmethod
    def supported_modalities(cls):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def select_parameter(self, modality, parameter_name, default_value):
        if parameter_name in self.modalities[modality]:
            return self.modalities[modality][parameter_name]
        else:
            return default_value

    @abstractmethod
    def get_buffer_shape(self, modality: str):
        raise NotImplementedError

    def get_buffer(self, modality: str):
        return np.zeros(self.get_buffer_shape(modality), dtype="float32")

    def get_shard_size(self, modality: str) -> int:
        return self.modalities[modality]["shard_size"]
