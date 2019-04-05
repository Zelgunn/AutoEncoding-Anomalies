import numpy as np
import cv2
from typing import Union, List, Dict, Any, Tuple

from datasets.modality_builders import ModalityBuilder, VideoReader


class VideoBuilder(ModalityBuilder):
    def __init__(self,
                 modalities: Union[str, List[str], Dict[str, Dict[str, Any]]],
                 video_reader: VideoReader,
                 default_frame_size: Union[Tuple[int, int], List[int], None]):
        super(VideoBuilder, self).__init__(modalities=modalities)

        self.video_reader = video_reader
        self.default_frame_size = default_frame_size

        self.frame_count = video_reader.frame_count
        if "flow" in self.modalities or "dog" in self.modalities:
            self.frame_count -= 1
            self.skip_first = True
        else:
            self.skip_first = False

    @classmethod
    def supported_modalities(cls):
        return ["raw_video", "flow", "dog"]

    def __iter__(self):
        shard: Dict[str, np.ndarray] = {modality: self.get_buffer(modality) for modality in self.modalities}
        previous_frame = None
        video_shard_size = self.get_shard_size("raw_video")

        # region Parameters
        # region Raw
        raw_frame_size = None

        if "raw_video" in self.modalities:
            raw_frame_size = self.get_frame_size("raw_video", none_if_reader_default=True)
        # endregion

        # region Flow
        flow_frame_size, farneback_params, flow_use_polar = None, None, None

        if "flow" in self.modalities:
            flow_frame_size = self.get_frame_size("flow", none_if_reader_default=True)
            farneback_params, flow_use_polar = self.get_flow_params()

        # endregion

        # region DoG
        dog_frame_size, dog_blurs = None, None
        if "dog" in self.modalities:
            dog_frame_size = self.get_frame_size("dog", none_if_reader_default=True)
            dog_blurs = self.get_dog_params()

        # endregion
        # endregion

        i = 0
        for frame in self.video_reader:
            frame = frame.astype("float64")

            if self.skip_first and previous_frame is None:
                previous_frame = frame
                continue

            video_index_in_shard = i % video_shard_size
            if "raw_video" in shard:
                raw_video = compute_raw(frame, raw_frame_size)
                shard["raw_video"][video_index_in_shard] = raw_video

            if "flow" in shard:
                shard["flow"][video_index_in_shard] = compute_flow(previous_frame, frame, farneback_params,
                                                                   flow_use_polar, flow_frame_size)

            if "dog" in shard:
                shard["dog"][video_index_in_shard] = compute_difference_of_gaussians(frame, dog_blurs, dog_frame_size)

            i += 1
            previous_frame = frame

            # region yield
            if i % video_shard_size == 0:
                yield shard

            if i == self.frame_count:
                remain_size = i % video_shard_size
                shard = {modality: shard[modality][:remain_size] for modality in shard}
                yield shard
            # endregion

    def get_buffer_shape(self, modality: str):
        frame_size = self.get_frame_size(modality)
        shard_size = self.get_shard_size(modality)

        if modality == "raw_video":
            return [shard_size, *frame_size, self.video_reader.frame_channels]
        elif modality == "flow":
            return [shard_size, *frame_size, 2]
        elif modality == "dog":
            dog_blurs = self.get_dog_params()
            return [shard_size, *frame_size, len(dog_blurs) - 1]
        else:
            raise ValueError

    def get_frame_size(self, modality: str, none_if_reader_default=False):
        if self.default_frame_size is not None:
            default_frame_size = self.default_frame_size
        else:
            default_frame_size = self.video_reader.frame_size

        frame_size = self.select_parameter(modality, "frame_size", default_frame_size)

        if none_if_reader_default and self.video_reader.frame_size == frame_size:
            frame_size = None

        return frame_size

    def get_frame_count(self, modality: str) -> int:
        return self.frame_count

    # region Flow
    def get_flow_params(self):
        farneback_params = {"pyr_scale": 0.5,
                            "levels": 3,
                            "winsize": 5,
                            "iterations": 5,
                            "poly_n": 5,
                            "poly_sigma": 1.2}
        farneback_params = {param_name: self.select_parameter("flow", param_name, farneback_params[param_name])
                            for param_name in farneback_params}
        flow_use_polar = self.select_parameter("flow", "use_polar", default_value=False)
        return farneback_params, flow_use_polar

    def pre_compute_flow_max(self) -> np.ndarray:
        farneback_params, flow_use_polar = self.get_flow_params()
        flow_frame_size = self.get_frame_size("flow", none_if_reader_default=True)

        previous_frame = None
        max_flow = 0.0
        for frame in self.video_reader:
            frame = frame.astype("float64")

            if previous_frame is None:
                previous_frame = frame
                continue

            flow = compute_flow(previous_frame, frame, farneback_params, flow_use_polar,
                                flow_frame_size)

            if flow_use_polar:
                max_flow = max(max_flow, np.max(flow[:, :, 0]))
            else:
                max_flow = max(max_flow, np.max(flow))

        if flow_use_polar:
            max_flow = np.array([max_flow, 360.0])
        else:
            max_flow = np.array(max_flow)

        return max_flow

    # endregion

    def get_dog_params(self):
        assert "dog" in self.modalities
        blurs = self.modalities["dog"]["blurs"] if "blurs" in self.modalities["dog"] else [2.0, 2.82, 4.0, 5.66, 8.0]
        return blurs


# region Compute modalities
def compute_raw(frame: np.ndarray,
                frame_size: Tuple[int, int]):
    raw = frame

    if frame_size is not None:
        raw = cv2.resize(raw, dsize=tuple(reversed(frame_size)))

    if raw.ndim == 2:
        raw = np.expand_dims(raw, axis=-1)

    return raw


# TODO : Compute flow from other sources (copy from preprocessed file, from model, ...)
def compute_flow(previous_frame: np.ndarray,
                 frame: np.ndarray,
                 farneback_params: Dict,
                 use_polar: bool,
                 frame_size: Tuple[int, int]):
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
        previous_frame = previous_frame.mean(axis=-1)

    flow: np.ndarray = cv2.calcOpticalFlowFarneback(prev=previous_frame, next=frame, flow=None,
                                                    flags=0, **farneback_params)

    absolute_flow = np.abs(flow)
    if np.min(absolute_flow) < 1e-20:
        flow[absolute_flow < 1e-20] = 0.0

    if use_polar:
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


def main():
    video_reader = VideoReader(r"..\datasets\ucsd\ped2\Test\Test001")
    video_builder = VideoBuilder(modalities={"raw_video": {},
                                             "flow": {"use_polar": True},
                                             "dog": {}
                                             },
                                 video_reader=video_reader,
                                 default_frame_size=(512, 512))

    for shard in video_builder:
        show_shard(shard)


if __name__ == "__main__":
    main()
