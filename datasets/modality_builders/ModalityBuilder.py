import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict

from modalities import Modality, ModalityCollection


class ModalityBuilder(ABC):
    def __init__(self,
                 shard_duration: float,
                 modalities: ModalityCollection):
        self.assert_supported_modalities(modalities)

        self.shard_duration = shard_duration
        self.modalities = modalities

    @classmethod
    @abstractmethod
    def supported_modalities(cls):
        raise NotImplementedError

    @classmethod
    def supports(cls, modality: Modality):
        return type(modality) in cls.supported_modalities()

    @classmethod
    def assert_supported_modalities(cls, modalities: ModalityCollection):
        if not all([cls.supports(modality) for modality in modalities]):
            unsupported_mods = [str(type(modality)) for modality in modalities if not cls.supports(modality)]
            raise NotImplementedError(",".join(unsupported_mods) + " are not supported.")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def get_shard_buffer(self) -> Dict[Type[Modality], np.ndarray]:
        shard_buffer = {type(modality): self.get_modality_buffer(modality)
                        for modality in self.modalities}
        return shard_buffer

    def get_modality_buffer(self, modality: Modality):
        return np.zeros(self.get_modality_buffer_shape(modality), dtype="float32")

    @abstractmethod
    def get_modality_buffer_shape(self, modality: Modality):
        raise NotImplementedError

    def get_modality_max_shard_size(self, modality: Modality):
        return int(np.ceil(self.shard_duration * modality.frequency))

    def get_initial_shard_sizes(self) -> Dict[Type[Modality], int]:
        shard_sizes = {type(modality): self.get_modality_initial_shard_size(modality)
                       for modality in self.modalities}
        return shard_sizes

    def get_modality_initial_shard_size(self, modality: Modality):
        return int(np.floor(self.shard_duration * modality.frequency))

    def get_modality_next_shard_size(self,
                                     modality: Modality,
                                     time: float):
        yielded_frame_count = int(np.floor(time * modality.frequency))

        total_frame_count_yielded_next_time = int(np.floor(modality.frequency * (time + self.shard_duration)))

        frame_count = self.get_modality_frame_count(modality)
        if total_frame_count_yielded_next_time > frame_count:
            total_frame_count_yielded_next_time = frame_count

        shard_size = total_frame_count_yielded_next_time - yielded_frame_count
        return shard_size

    def get_next_shard_sizes(self, time: float) -> Dict[Type[Modality], int]:
        shard_sizes = {type(modality): self.get_modality_next_shard_size(modality, time)
                       for modality in self.modalities}
        return shard_sizes

    @staticmethod
    def extract_shard(shard_buffer, shard_sizes):
        shard = {modality: shard_buffer[modality][:shard_sizes[modality]]
                 for modality in shard_buffer}
        return shard

    @abstractmethod
    def get_modality_frame_count(self, modality: Modality) -> int:
        raise NotImplementedError

    def get_shard_count(self) -> int:
        modality_durations = []

        for modality in self.modalities:
            frame_count = self.get_modality_frame_count(modality)
            modality_duration = frame_count / modality.frequency
            modality_durations.append(modality_duration)

        min_duration = min(*modality_durations)
        shard_count = int(np.ceil(min_duration / self.shard_duration))

        return shard_count
