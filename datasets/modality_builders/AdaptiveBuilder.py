import cv2
import numpy as np
from typing import Union, List, Dict, Any, Tuple, Type

from datasets.modality_builders import ModalityBuilder, VideoBuilder, AudioBuilder
from datasets.data_readers import VideoReader, AudioReader


class AdaptiveBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 modalities: Union[str, List[str], Dict[str, Dict[str, Any]]],
                 video_source: Union[VideoReader, str, cv2.VideoCapture, np.ndarray, List[str]],
                 video_frame_size: Tuple[int, int],
                 audio_source: Union[AudioReader, str, np.ndarray]):
        super(AdaptiveBuilder, self).__init__(shard_duration=shard_duration,
                                              modalities=modalities)

        self.builders: List[ModalityBuilder] = []

        # region Video
        video_modalities = {modality: modalities[modality] for modality in modalities
                            if modality in VideoBuilder.supported_modalities()}

        if len(video_modalities) > 0:
            if not isinstance(video_source, VideoReader):
                video_reader = VideoReader(video_source)
            else:
                video_reader = video_source

            video_builder = VideoBuilder(shard_duration=shard_duration,
                                         modalities=video_modalities,
                                         video_reader=video_reader,
                                         default_frame_size=video_frame_size)
            self.builders.append(video_builder)
        # endregion

        # region Audio
        audio_modalities = {modality: modalities[modality] for modality in modalities
                            if modality in AudioBuilder.supported_modalities()}

        if len(audio_modalities) > 0:
            if not isinstance(video_source, VideoReader):
                audio_reader = AudioReader(audio_source)
            else:
                audio_reader = audio_source

            audio_builder = AudioBuilder(shard_duration=shard_duration,
                                         modalities=audio_modalities,
                                         audio_reader=audio_reader)
            self.builders.append(audio_builder)
        # endregion

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

    def get_modality_buffer_shape(self, modality: str):
        for builder in self.builders:
            if modality in builder.modalities:
                return builder.get_modality_buffer_shape(modality)
        raise ValueError("Could not find a builder with modality `{}`".format(modality))

    def get_modality_frame_count(self, modality: str):
        for builder in self.builders:
            if modality in builder.modalities:
                return builder.get_modality_frame_count(modality)
        raise ValueError("Could not find a builder with modality `{}`".format(modality))
