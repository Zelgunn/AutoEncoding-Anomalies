import numpy as np
from typing import List, Union, Any, Dict, Type

from modalities import Modality, ModalityCollection, RawAudio, MelSpectrogram
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers import AudioReader


class AudioBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 source_frequency: Union[int, float],
                 modalities: ModalityCollection,
                 audio_reader: Union[AudioReader, Any]):
        super(AudioBuilder, self).__init__(shard_duration=shard_duration,
                                           source_frequency=source_frequency,
                                           modalities=modalities)

        if not isinstance(audio_reader, AudioReader):
            audio_reader = AudioReader(audio_reader)
        else:
            audio_reader = audio_reader

        self.reader = audio_reader

    @classmethod
    def supported_modalities(cls):
        return [RawAudio, MelSpectrogram]

    def check_shard(self, frames: np.ndarray) -> bool:
        return frames.shape[0] > (self.source_frequency * 0.001)

    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        shard: Dict[Type[Modality], np.ndarray] = {}

        if RawAudio in self.modalities:
            shard[RawAudio] = frames

        if MelSpectrogram in self.modalities:
            modality: MelSpectrogram = self.modalities[MelSpectrogram]
            shard[MelSpectrogram] = modality.wave_to_mel_spectrogram(frames, self.source_frequency)

        return shard

    def get_buffer_shape(self) -> List[int]:
        max_shard_size = self.get_source_max_shard_size()
        return [max_shard_size, self.reader.channels_count]

    @property
    def source_frame_count(self):
        return self.reader.frame_count
