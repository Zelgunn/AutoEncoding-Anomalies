import numpy as np
import cv2
import os
from typing import List

from datasets import PartiallyLoadableDatabase
from data_preprocessors import DataPreprocessor


class SubwayDatabase(PartiallyLoadableDatabase):
    def __init__(self,
                 database_path: str,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        super(SubwayDatabase, self).__init__(database_path=database_path,
                                             train_preprocessors=train_preprocessors,
                                             test_preprocessors=test_preprocessors)
        self._base_name = "Subway_Exit" if "exit" in database_path.lower() else "Subway_Entrance"
        self.video_capture = None
        self.frame_count = None
        self.frame_height = None
        self.frame_width = None
        self.fourcc = None
        self.fps = None

    def on_build_shard_begin(self, height, width):
        video_filename = "{0}_{1}x{2}.avi".format(self.base_name, height, width)
        video_filepath = os.path.join(self.database_path, video_filename)
        if not os.path.exists(video_filepath):
            self.save_resized_video(height, width)
        self.open_video(video_filepath)

    def build_shards_iterator(self, shard_size, skip=0):
        if self.video_capture is None:
            self.open_video(self.base_video_filepath)

        split = int(self.fps * 300)  # 5 minutes
        print("Defaulting to {0} frames for Subway Exit".format(split))
        for shard in self.build_subset_shards_iterator(shard_size, skip, 0, split):
            yield "train", shard
        for shard in self.build_subset_shards_iterator(shard_size, skip, split, -1):
            yield "test", shard

    @property
    def base_name(self):
        return self._base_name

    @property
    def base_video_filepath(self):
        return os.path.join(self.database_path, self.base_name + ".avi")

    def open_video(self, video_filepath: str):
        if self.video_capture is not None:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(video_filepath)

        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

    def build_subset_shards_iterator(self, shard_size, skip=0, start=0, end=-1):
        start = start if start >= 0 else self.frame_count + start + 1
        end = end if end >= 0 else self.frame_count + end + 1
        assert start < end <= self.frame_count

        frame_count = (end - start) // (skip + 1)
        shard_count = int(np.ceil(frame_count / shard_size))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in range(shard_count):
            if i == (shard_count - 1):
                shard_size = frame_count - shard_size * i
            shard = np.empty(shape=[shard_size, self.frame_height, self.frame_width])
            for j in range(shard_size):
                ret, frame = self.video_capture.read()
                shard[j] = np.mean(frame, axis=-1)
                if skip > 0:
                    next_index = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) + skip
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(next_index))
            shard = shard / 255.0
            yield shard

    def save_resized_video(self, height, width):
        if self.video_capture is None:
            self.open_video(self.base_video_filepath)

        video_filepath = os.path.join(self.database_path, "Subway_Exit_{0}x{1}.avi".format(height, width))
        frame_size = (height, width)
        video_writer = cv2.VideoWriter(video_filepath, self.fourcc, self.fps, frame_size)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            print("\r{0}/{1}".format(i + 1, self.frame_count), end='')
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, dsize=frame_size)
            video_writer.write(frame)
        print('', end=os.linesep)


# path = os.path.normpath("../datasets/subway/exit")
# subway_database = SubwayDatabase(path)
# prepared_resolutions = [
#     [192, 256],
#     [96, 128],
#     [48, 64],
#     [24, 32]
# ]
# subway_database.prepare_resolutions(prepared_resolutions, shard_size=2000, skip=4)
# exit()
