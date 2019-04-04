import numpy as np
import cv2
from PIL import Image
import os
from enum import IntEnum
from typing import Union, List


class VideoReaderMode(IntEnum):
    CV_VIDEO_CAPTURE = 0,
    NP_ARRAY = 1,
    IMAGE_COLLECTION = 2


class VideoReader(object):
    def __init__(self,
                 video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                 mode: VideoReaderMode = None):

        if mode is None:
            self.mode = infer_video_reader_mode(video_source)
        else:
            assert mode == infer_video_reader_mode(video_source)
            self.mode = mode

        self.video_source = video_source

        self.video_capture: cv2.VideoCapture = None
        self.video_array: np.array = None
        self.image_collection: List[str] = None

        # region Select & set container
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            if isinstance(video_source, str):
                self.video_capture = cv2.VideoCapture(video_source)
            else:
                self.video_capture = video_source
        elif self.mode == VideoReaderMode.NP_ARRAY:
            if isinstance(video_source, str):
                self.video_array = np.load(video_source, mmap_mode="r")
            else:
                self.video_array = video_source
        else:
            if isinstance(video_source, str):
                images_names = os.listdir(video_source)
                self.image_collection = [os.path.join(video_source, image_name) for image_name in images_names]
                self.image_collection = sorted(self.image_collection)
            else:
                self.image_collection = video_source
        # endregion

    def __iter__(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(self.frame_count):
            if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
                ret, frame = self.video_capture.read()
            elif self.mode == VideoReaderMode.NP_ARRAY:
                frame = self.video_array[i]
            else:
                frame = Image.open(self.image_collection[i])
                frame = np.array(frame)

            yield frame

    # region Properties
    @property
    def frame_count(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return len(self.video_array)
        else:
            return len(self.image_collection)

    @property
    def frame_height(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.height

    @property
    def frame_width(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[2]
        else:
            frame = Image.open(self.image_collection[0])
            return frame.width

    @property
    def frame_channels(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return 3
        elif self.mode == VideoReaderMode.NP_ARRAY:
            if self.video_array.ndim == 4:
                return self.video_array.shape[3]
            else:
                return 1
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            if frame.ndim == 3:
                return frame.shape[2]
            else:
                return 1

    @property
    def frame_shape(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width, 3]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:]
        else:
            frame = Image.open(self.image_collection[0])
            frame = np.array(frame)
            return frame.shape

    @property
    def frame_size(self):
        if self.mode == VideoReaderMode.CV_VIDEO_CAPTURE:
            return [self.frame_height, self.frame_width]
        elif self.mode == VideoReaderMode.NP_ARRAY:
            return self.video_array.shape[1:3]
        else:
            frame = Image.open(self.image_collection[0])
            return tuple(reversed(frame.size))
    # endregion


def infer_video_reader_mode(video_source: Union[str, cv2.VideoCapture, np.ndarray, List[str]]):
    if isinstance(video_source, cv2.VideoCapture):
        return VideoReaderMode.CV_VIDEO_CAPTURE

    elif isinstance(video_source, np.ndarray):
        assert video_source.ndim == 3 or video_source.ndim == 4
        return VideoReaderMode.NP_ARRAY

    elif isinstance(video_source, list):
        assert all([isinstance(element, str) for element in video_source])
        assert all([os.path.isfile(element) for element in video_source])
        return VideoReaderMode.IMAGE_COLLECTION

    elif not isinstance(video_source, str):
        raise ValueError("\'video_source\' must either be a string, a VideoCapture, a ndarray or a list of strings")

    elif os.path.isdir(video_source):
        return VideoReaderMode.IMAGE_COLLECTION

    elif os.path.isfile(video_source):
        if ".npy" in video_source or ".npz" in video_source:
            return VideoReaderMode.NP_ARRAY
        else:
            return VideoReaderMode.CV_VIDEO_CAPTURE

    else:
        raise ValueError("\'video_source\' : {} does not exist.".format(video_source))


def main():
    # TODO : Filter image collection inputs with supported image extensions
    video_reader = VideoReader(r"..\datasets\ucsd\ped2\Test\Test001_gt")
    print(video_reader.image_collection)
    for frame in video_reader:
        cv2.imshow("frame", frame)
        cv2.waitKey(40)


if __name__ == "__main__":
    main()