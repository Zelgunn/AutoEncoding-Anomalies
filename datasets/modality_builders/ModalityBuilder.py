import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any


class ModalityBuilder(ABC):
    def __init__(self,
                 shard_duration: float,
                 modalities: Dict[str, Dict[str, Any]]):
        for modality in modalities:
            assert "frequency" in modalities[modality], "`frequency` not in modality `{}`".format(modality)
        self.shard_duration = shard_duration
        self.modalities = modalities

    @classmethod
    @abstractmethod
    def supported_modalities(cls):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    # region Modalities helper functions
    def select_parameter(self, modality, parameter_name, default_value):
        if parameter_name in self.modalities[modality]:
            return self.modalities[modality][parameter_name]
        else:
            return default_value

    def get_modality_buffer(self, modality: str):
        return np.zeros(self.get_modality_buffer_shape(modality), dtype="float32")

    @abstractmethod
    def get_modality_buffer_shape(self, modality: str):
        raise NotImplementedError

    def get_modality_max_shard_size(self, modality: str):
        return int(np.ceil(self.shard_duration * self.get_modality_frequency(modality)))

    def get_modality_frequency(self, modality: str) -> float:
        return self.modalities[modality]["frequency"]

    def get_modality_initial_shard_size(self, modality: str):
        return int(np.floor(self.shard_duration * self.get_modality_frequency(modality)))

    def get_modality_next_shard_size(self,
                                     modality: str,
                                     time: float):
        frequency = self.get_modality_frequency(modality)
        yielded_frame_count = int(np.floor(time * frequency))

        total_frame_count_yielded_next_time = int(np.floor(frequency * (time + self.shard_duration)))

        frame_count = self.get_modality_frame_count(modality)
        if total_frame_count_yielded_next_time > frame_count:
            total_frame_count_yielded_next_time = frame_count

        shard_size = total_frame_count_yielded_next_time - yielded_frame_count
        return shard_size

    @abstractmethod
    def get_modality_frame_count(self, modality: str) -> int:
        raise NotImplementedError

    # endregion

    def get_shard_count(self) -> int:
        modality_durations = []

        for modality in self.modalities:
            modality_duration = self.get_modality_frame_count(modality) / self.get_modality_frequency(modality)
            modality_durations.append(modality_duration)

        min_duration = min(*modality_durations)
        shard_count = int(np.ceil(min_duration / self.shard_duration))

        return shard_count
