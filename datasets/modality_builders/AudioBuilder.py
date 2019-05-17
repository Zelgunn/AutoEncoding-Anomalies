from typing import List

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
        shard_buffer = self.get_shard_buffer()

        # TODO : Make function for this (reference_mod)
        reference_mod = None
        for audio_mod in [RawAudio, MFCCs]:
            if audio_mod in shard_buffer:
                reference_mod = audio_mod
                break

        shard_sizes = self.get_initial_shard_sizes()
        reference_shard_size = shard_sizes[reference_mod]

        i = 0
        time = 0.0

        for frame in self.audio_reader:
            if RawAudio in shard_buffer:
                shard_buffer[RawAudio][i] = frame

            i += 1

            if (i % reference_shard_size) == 0:
                shard = self.extract_shard(shard_buffer, shard_sizes)
                yield shard

                # region Prepare next shard
                time += self.shard_duration
                shard_sizes = self.get_next_shard_sizes(time)
                reference_shard_size = shard_sizes[reference_mod]
                i = 0
                # endregion

    def get_modality_buffer_shape(self, modality: Modality) -> List[int]:
        max_shard_size = self.get_modality_max_shard_size(modality)

        if isinstance(modality, RawAudio):
            return [max_shard_size, self.audio_reader.channels_count]
        else:
            raise NotImplementedError(modality.id())

    def get_modality_frame_count(self, modality: Modality) -> int:
        if isinstance(modality, RawAudio):
            return self.audio_reader.frame_count
        else:
            raise NotImplementedError(modality.id())
