from typing import List, Union, Type

from modalities import Modality, ModalityCollection, RawAudio, MFCCs
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers import AudioReader


class AudioBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 modalities: ModalityCollection,
                 audio_reader: AudioReader):
        super(AudioBuilder, self).__init__(shard_duration=shard_duration,
                                           modalities=modalities)

        self.audio_reader = audio_reader

    @classmethod
    def supported_modalities(cls):
        return [RawAudio, MFCCs]

    def __iter__(self):
        raise NotImplementedError

    def get_modality_buffer_shape(self, modality: Modality):
        raise NotImplementedError

    def get_modality_frame_count(self, modality: Modality) -> int:
        raise NotImplementedError
