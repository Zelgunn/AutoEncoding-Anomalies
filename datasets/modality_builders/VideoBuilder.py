import numpy as np
import cv2
from typing import Union, List, Dict, Optional, Tuple

from modalities import Modality, ModalityCollection, RawVideo, OpticalFlow, DoG
from datasets.modality_builders import ModalityBuilder
from datasets.data_readers import VideoReader


class VideoBuilder(ModalityBuilder):
    def __init__(self,
                 shard_duration: float,
                 modalities: ModalityCollection,
                 video_reader: VideoReader,
                 default_frame_size: Union[Tuple[int, int], List[int], None]):
        super(VideoBuilder, self).__init__(shard_duration=shard_duration,
                                           modalities=modalities)

        self.video_reader = video_reader
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

    def __iter__(self):
        shard_buffer = self.get_shard_buffer()
        previous_frame = None
        time = 0.0

        # region Parameters
        frame_size = self.get_frame_size(none_if_reader_default=True)

        optical_flow: Optional[OpticalFlow] = None
        dog: Optional[DoG] = None

        if OpticalFlow in self.modalities:
            optical_flow: OpticalFlow = self.modalities[OpticalFlow]

        if DoG in self.modalities:
            dog: DoG = self.modalities[DoG]
        # endregion

        # region Compute initial shard size
        reference_mod = None

        for video_mod in [RawVideo, OpticalFlow, DoG]:
            if video_mod in shard_buffer:
                reference_mod = video_mod
                break

        shard_sizes = self.get_initial_shard_sizes()
        video_shard_size = shard_sizes[reference_mod]
        # endregion

        i = 0
        for frame in self.video_reader:
            frame = frame.astype("float64")

            # region Skip first
            if self.skip_first and previous_frame is None:
                previous_frame = frame
                continue
            # endregion

            # region Compute video modalities
            if RawVideo in shard_buffer:
                raw_video = compute_raw(frame, frame_size)
                shard_buffer[RawVideo][i] = raw_video

            if OpticalFlow in shard_buffer:
                shard_buffer[OpticalFlow][i] = compute_flow(previous_frame, frame, optical_flow, frame_size)

            if DoG in shard_buffer:
                shard_buffer[DoG][i] = compute_difference_of_gaussians(frame, dog.blurs, frame_size)
            # endregion

            # region Iterate i
            previous_frame = frame
            i += 1
            # endregion

            # region Yield shard
            if i % video_shard_size == 0:
                shard = self.extract_shard(shard_buffer, shard_sizes)
                yield shard

                # region Prepare next shard
                time += self.shard_duration
                shard_sizes = self.get_next_shard_sizes(time)
                video_shard_size = shard_sizes[reference_mod]
                i = 0
                # endregion
            # endregion

    def get_modality_buffer_shape(self, modality: Modality):
        frame_size = self.get_frame_size()
        max_shard_size = self.get_modality_max_shard_size(modality)

        if isinstance(modality, RawVideo):
            return [max_shard_size, *frame_size, self.video_reader.frame_channels]
        elif isinstance(modality, OpticalFlow):
            return [max_shard_size, *frame_size, 2]
        elif isinstance(modality, DoG):
            return [max_shard_size, *frame_size, len(modality.blurs) - 1]
        else:
            raise ValueError("{} of type {} is not supported.".format(modality, type(modality)))

    def get_frame_size(self, none_if_reader_default=False):
        if self.default_frame_size is not None:
            return self.default_frame_size
        else:
            if none_if_reader_default:
                return None
            else:
                return self.video_reader.frame_size

    def get_modality_frame_count(self, modality: Modality) -> int:
        return self.frame_count

    def pre_compute_flow_max(self) -> np.ndarray:
        optical_flow: OpticalFlow = self.modalities[OpticalFlow]
        flow_frame_size = self.get_frame_size(none_if_reader_default=True)

        previous_frame = None
        max_flow = 0.0
        for frame in self.video_reader:
            frame = frame.astype("float64")

            if previous_frame is None:
                previous_frame = frame
                continue

            flow = compute_flow(previous_frame, frame, optical_flow, flow_frame_size)

            if optical_flow.use_polar:
                max_flow = max(max_flow, np.max(flow[:, :, 0]))
            else:
                max_flow = max(max_flow, np.max(flow))

        if optical_flow.use_polar:
            max_flow = np.array([max_flow, 360.0])
        else:
            max_flow = np.array(max_flow)

        return max_flow


# region Compute modalities
def compute_raw(frame: np.ndarray,
                frame_size: Tuple[int, int]):
    raw = frame

    if frame_size is not None:
        raw = cv2.resize(raw, dsize=tuple(reversed(frame_size)))

    if raw.ndim == 2:
        raw = np.expand_dims(raw, axis=-1)

    return raw


# TODO : Compute flow with other tools (copy from preprocessed file, from model, ...)
def compute_flow(previous_frame: np.ndarray,
                 frame: np.ndarray,
                 optical_flow: OpticalFlow,
                 frame_size: Tuple[int, int]):
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
        previous_frame = previous_frame.mean(axis=-1)

    flow: np.ndarray = cv2.calcOpticalFlowFarneback(prev=previous_frame, next=frame, flow=None,
                                                    flags=0, **optical_flow.farneback_params)

    absolute_flow = np.abs(flow)
    if np.min(absolute_flow) < 1e-20:
        flow[absolute_flow < 1e-20] = 0.0

    if optical_flow.use_polar:
        flow = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
        flow = np.stack(flow, axis=-1)

    if frame_size is not None:
        flow = cv2.resize(flow, dsize=tuple(reversed(frame_size)))

    return flow


def compute_difference_of_gaussians(frame: np.ndarray,
                                    blurs: np.ndarray or List[int],
                                    frame_size: Tuple[int, int]):
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            frame = np.squeeze(frame, axis=2)
        else:
            frame = np.mean(frame, axis=2)

    frames = [None] * len(blurs)
    for i in range(len(blurs)):
        frames[i] = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=blurs[i])
        if frame_size is not None:
            frames[i] = cv2.resize(frames[i], dsize=tuple(reversed(frame_size)))

    deltas = [None] * (len(blurs) - 1)

    for i in range(len(deltas)):
        deltas[i] = np.abs(frames[i] - frames[i + 1])

    deltas = np.stack(deltas, axis=-1)

    return deltas


# endregion


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
