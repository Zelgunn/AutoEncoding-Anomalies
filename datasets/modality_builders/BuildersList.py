import cv2
import numpy as np
from typing import Union, List, Tuple, Type

from modalities import Modality, ModalityCollection
from datasets.modality_builders import ModalityBuilder, VideoBuilder, AudioBuilder
from datasets.data_readers import VideoReader, AudioReader


class BuildersList(object):
    def __init__(self, builders: List[ModalityBuilder]):
        modalities = []
        for builder in builders:
            for modality in builder.modalities:
                modalities.append(modality)

        self.builders: List[ModalityBuilder] = builders

    @staticmethod
    def supported_builders() -> List[Type[ModalityBuilder]]:
        return [VideoBuilder, AudioBuilder]

    @classmethod
    def supported_modalities(cls):
        return [modality
                for builder_class in cls.supported_builders()
                for modality in builder_class.supported_modalities()]

    def __iter__(self):
        builders_iterator = zip(*self.builders)
        for partial_shards in builders_iterator:
            shard = {modality: partial_shard[modality]
                     for partial_shard in partial_shards
                     for modality in partial_shard}

            yield shard

    def get_shard_count(self) -> int:
        min_shard_count = None

        for builder in self.builders:
            total_duration = builder.source_frame_count / builder.source_frequency
            shard_count = total_duration / builder.shard_duration
            if min_shard_count is None:
                min_shard_count = shard_count
            else:
                min_shard_count = min(min_shard_count, shard_count)

        min_shard_count = int(np.ceil(min_shard_count))
        return min_shard_count
