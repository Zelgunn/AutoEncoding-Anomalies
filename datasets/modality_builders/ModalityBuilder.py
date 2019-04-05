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
        shard_size = self.modalities[modality]["shard_size"]

        names_met = []
        while isinstance(shard_size, str):
            names_met.append(shard_size)
            shard_size = self.modalities[shard_size]["shard_size"]
            if shard_size in names_met:
                raise ValueError("Cyclic reference in shard size : " + " -> ".join(names_met) + " -> " + shard_size)

        return shard_size

    @abstractmethod
    def get_frame_count(self, modality: str) -> int:
        raise NotImplementedError

    def get_shard_count(self) -> int:
        min_count = None

        for modality in self.modalities:
            count = int(np.ceil(self.get_frame_count(modality) / self.get_shard_size(modality)))
            if min_count is None:
                min_count = count
            else:
                min_count = min(min_count, count)

        return min_count
