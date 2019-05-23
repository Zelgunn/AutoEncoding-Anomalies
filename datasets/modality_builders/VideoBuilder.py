import numpy as np
import cv2
from typing import Union, List, Optional, Tuple, Any, Dict, Type

from modalities import Modality, ModalityCollection, RawVideo, OpticalFlow, DoG
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
        return [RawVideo, OpticalFlow, DoG]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.default_frame_size is not None:
            if tuple(self.default_frame_size) != frame.shape[:2]:
                frame = cv2.resize(frame, dsize=tuple(reversed(self.default_frame_size)))
        return frame

    def check_shard(self, frames: np.ndarray) -> bool:
        return frames.shape[0] > 1

    def process_shard(self, frames: np.ndarray) -> Dict[Type[Modality], np.ndarray]:
        shard: Dict[Type[Modality], np.ndarray] = {}
        frames = frames.astype(np.float64)

        if RawVideo in self.modalities:
            shard[RawVideo] = frames

        if OpticalFlow in self.modalities:
            optical_flow: OpticalFlow = self.modalities[OpticalFlow]
            shard[OpticalFlow] = optical_flow.compute_flow(frames, self.default_frame_size)

        if DoG in self.modalities:
            dog: DoG = self.modalities[DoG]
            shard[DoG] = dog.compute_difference_of_gaussians(frames, self.default_frame_size)

        return shard

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

    def get_modality_frame_count(self, modality: Modality) -> int:
        return self.frame_count

    @property
    def source_frame_count(self):
        return self.reader.frame_count

    def pre_compute_flow_max(self) -> np.ndarray:
        optical_flow: OpticalFlow = self.modalities[OpticalFlow]
        flow_frame_size = self.get_frame_size(none_if_reader_default=True)

        previous_frame = None
        max_flow = 0.0
        for frame in self.reader:
            frame = frame.astype("float64")

            if previous_frame is None:
                previous_frame = frame
                continue

            flow = optical_flow.compute_flow_frame(previous_frame, frame, flow_frame_size)

            if optical_flow.use_polar:
                max_flow = max(max_flow, np.max(flow[:, :, 0]))
            else:
                max_flow = max(max_flow, np.max(flow))

        if optical_flow.use_polar:
            max_flow = np.array([max_flow, 360.0])
        else:
            max_flow = np.array(max_flow)

        return max_flow


def show_shard(shard):
    for i in range(shard["raw_video"].shape[0]):
        for modality in shard:
            frame: np.ndarray = shard[modality][i]

            if frame.ndim < 2 or frame.dtype not in ["float32", "float64"]:
                continue

            if frame.shape[-1] == 2:
                frame = np.stack([frame[:, :, 1], np.ones_like(frame[:, :, 1]), frame[:, :, 0] * 0.5], axis=-1)
                frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
                _, frame = cv2.threshold(frame, thresh=0.25, maxval=255.0, type=cv2.THRESH_TOZERO)

            if frame.shape[-1] > 4:
                frame = frame[:, :, :4]

            cv2.imshow(modality, frame)
        cv2.waitKey(40)
