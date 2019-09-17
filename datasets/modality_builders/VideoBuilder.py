import numpy as np
import cv2
from typing import Union, List, Tuple, Any, Dict, Type

from modalities import Modality, ModalityCollection, RawVideo, Faces, OpticalFlow, DoG, Landmarks
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers import VideoReader


class VideoBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 source_frequency: Union[int, float],
                 modalities: ModalityCollection,
                 video_reader: Union[VideoReader, Any],
                 default_frame_size: Union[Tuple[int, int], List[int], None]):

        super(VideoBuilder, self).__init__(shard_duration=shard_duration,
                                           source_frequency=source_frequency,
                                           modalities=modalities)

        if not isinstance(video_reader, VideoReader):
            video_reader = VideoReader(video_reader)
        else:
            video_reader = video_reader

        self.reader = video_reader
        self.default_frame_size = default_frame_size

        self.frame_count = video_reader.frame_count
        if OpticalFlow in self.modalities or DoG in self.modalities:
            self.frame_count -= 1
            self.skip_first = True
        else:
            self.skip_first = False

    @classmethod
    def supported_modalities(cls):
        return [RawVideo, Faces, OpticalFlow, DoG, Landmarks]

    # region Frame/Shard processing
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.default_frame_size is not None:
            if tuple(self.default_frame_size) != frame.shape[:2]:
                frame = cv2.resize(frame, dsize=tuple(reversed(self.default_frame_size)))
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
        return frame

    def check_shard(self, frames: np.ndarray) -> bool:
        return frames.shape[0] > 1

    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        shard: Dict[Type[Modality], np.ndarray] = {}
        frames = frames.astype(np.float64)

        if RawVideo in self.modalities:
            shard[RawVideo] = frames

        if Faces in self.modalities:
            faces: Faces = self.modalities[Faces]
            shard[Faces] = faces.compute_faces(frames, self.default_frame_size)

        if OpticalFlow in self.modalities:
            optical_flow: OpticalFlow = self.modalities[OpticalFlow]
            shard[OpticalFlow] = optical_flow.compute_flow(frames, self.default_frame_size)

        if DoG in self.modalities:
            dog: DoG = self.modalities[DoG]
            shard[DoG] = dog.compute_difference_of_gaussians(frames, self.default_frame_size)

        if Landmarks in self.modalities:
            landmarks: Landmarks = self.modalities[Landmarks]
            shard[Landmarks] = landmarks.compute_landmarks(frames, use_other_if_fail=True)

        return shard

    # endregion

    def get_buffer_shape(self) -> List[int]:
        frame_size = self.get_frame_size()
        max_shard_size = self.get_source_max_shard_size()
        return [max_shard_size, *frame_size, self.reader.frame_channels]

    def get_frame_size(self, none_if_reader_default=False):
        if self.default_frame_size is not None:
            return self.default_frame_size
        else:
            if none_if_reader_default:
                return None
            else:
                return self.reader.frame_size

    @property
    def source_frame_count(self):
        return self.reader.frame_count
