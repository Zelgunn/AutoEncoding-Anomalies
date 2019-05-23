import numpy as np
from abc import ABC, abstractmethod
from typing import Type, Dict, Union, Iterable, Optional

from modalities import Modality, ModalityCollection, RawVideo, RawAudio, DoG, OpticalFlow, MelSpectrogram

EPSILON = 1e-5


# TODO : Remove unused methods
class ModalityBuilder(ABC):
    def __init__(self,
                 shard_duration: float,
                 source_frequency: Union[int, float],
                 modalities: ModalityCollection):
        modalities = self.filter_supported_modalities(modalities)

        self.shard_duration = shard_duration
        self.source_frequency = source_frequency

        self.modalities = modalities
        self.reader: Optional[Iterable] = None

    @classmethod
    @abstractmethod
    def supported_modalities(cls):
        raise NotImplementedError

    @classmethod
    def supports(cls, modality: Modality) -> bool:
        if not isinstance(modality, Modality):
            raise ValueError("`modality` is not a Modality, got type {}.".format(type(modality)))
        return type(modality) in cls.supported_modalities()

    @classmethod
    def supports_any(cls, modalities: ModalityCollection) -> bool:
        return any([cls.supports(modality) for modality in modalities])

    @classmethod
    def filter_supported_modalities(cls, modalities: ModalityCollection) -> ModalityCollection:
        filtered_modalities = []
        for modality in modalities:
            if cls.supports(modality):
                filtered_modalities.append(modality)
        return ModalityCollection(filtered_modalities)

    @classmethod
    def assert_supported_modalities(cls, modalities: ModalityCollection):
        if not all([cls.supports(modality) for modality in modalities]):
            unsupported_mods = [str(type(modality)) for modality in modalities if not cls.supports(modality)]
            unsupported_mods_str = ",".join(unsupported_mods)
            raise NotImplementedError("{} are not supported by {}.".format(unsupported_mods_str, cls.__name__))

    # @abstractmethod
    # def __iter__(self):
    #     raise NotImplementedError

    def __iter__(self):
        shard_buffer = self.get_shard_buffer()
        source_shard_size = self.get_source_initial_shard_size()

        i = 0
        time = 0.0

        if self.reader is None:
            raise ValueError("You must provide a reader for the builder")

        for frame in self.reader:
            shard_buffer[i] = self.process_frame(frame)

            i += 1

            if (i % source_shard_size) == 0:
                frames = shard_buffer[:source_shard_size]
                if self.check_shard(frames):
                    shard = self.process_shard(frames)
                    yield shard

                time += self.shard_duration
                source_shard_size = self.get_source_next_shard_size(time)
                i = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    @abstractmethod
    def check_shard(self, frames: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        raise NotImplementedError

    def get_shard_buffer(self) -> np.ndarray:
        return np.zeros(self.get_buffer_shape(), dtype="float32")

    @abstractmethod
    def get_buffer_shape(self):
        raise NotImplementedError

    def compute_shard_size(self, modality: Modality, source_shard_size: int) -> int:
        if isinstance(modality, RawVideo) or isinstance(modality, RawAudio):
            return source_shard_size
        elif isinstance(modality, OpticalFlow) or isinstance(modality, DoG):
            return source_shard_size - 1
        elif isinstance(modality, MelSpectrogram):
            return modality.get_output_frame_count(source_shard_size, self.source_frequency)
        else:
            raise NotImplementedError(modality.id())

    # region Max shard size
    def get_source_max_shard_size(self) -> int:
        return int(np.ceil(self.shard_duration * self.source_frequency - EPSILON))

    def get_modality_max_shard_size(self, modality: Modality) -> int:
        source_shard_size = self.get_source_max_shard_size()
        return self.compute_shard_size(modality, source_shard_size)
    # endregion

    # region Initial shard size
    def get_initial_shard_sizes(self) -> Dict[Type[Modality], int]:
        shard_sizes = {type(modality): self.get_modality_initial_shard_size(modality)
                       for modality in self.modalities}
        return shard_sizes

    def get_source_initial_shard_size(self) -> int:
        return int(np.floor(self.shard_duration * self.source_frequency + EPSILON))

    def get_modality_initial_shard_size(self, modality: Modality):
        source_shard_size = self.get_source_initial_shard_size()
        return self.compute_shard_size(modality, source_shard_size)
    # endregion

    # region Next shard size
    def get_source_next_shard_size(self, time: float):
        yielded_frame_count = int(np.floor(time * self.source_frequency + EPSILON))

        total_frame_count_yielded_next_time = self.source_frequency * (time + self.shard_duration) + EPSILON
        total_frame_count_yielded_next_time = int(np.floor(total_frame_count_yielded_next_time))

        if total_frame_count_yielded_next_time > self.source_frame_count:
            total_frame_count_yielded_next_time = self.source_frame_count

        shard_size = total_frame_count_yielded_next_time - yielded_frame_count

        return shard_size

    def get_next_shard_sizes(self, time: float) -> Dict[Type[Modality], int]:
        shard_sizes = {type(modality): self.get_source_next_shard_size(time)
                       for modality in self.modalities}
        return shard_sizes
    # endregion

    @staticmethod
    def extract_shard(shard_buffer, shard_sizes):
        shard = {modality: shard_buffer[modality][:shard_sizes[modality]]
                 for modality in shard_buffer}
        return shard

    @abstractmethod
    def get_modality_frame_count(self, modality: Modality) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def source_frame_count(self):
        raise NotImplementedError

    def get_shard_count(self) -> int:
        total_duration = self.source_frame_count / self.source_frequency
        shard_count = total_duration / self.shard_duration
        shard_count = int(np.ceil(shard_count))

        return shard_count
