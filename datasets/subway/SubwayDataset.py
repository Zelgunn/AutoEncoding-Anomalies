import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
from typing import List

from datasets import Dataset
from data_preprocessors import DataPreprocessor


class SubwayDataset(Dataset):
    def __init__(self,
                 dataset_path: str = None,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.video_capture = None
        self.frame_count = None
        self.frame_height = None
        self.frame_width = None
        self.fourcc = None
        self.fps = None
        super(SubwayDataset, self).__init__(dataset_path=dataset_path,
                                            data_preprocessors=data_preprocessors,
                                            batch_size=batch_size,
                                            epoch_length=epoch_length,
                                            shuffle_on_epoch_end=shuffle_on_epoch_end,
                                            **kwargs)

    def load(self, dataset_path: str, **kwargs):
        super(SubwayDataset, self).load(dataset_path, **kwargs)
        video_filepath = os.path.join(dataset_path, "Subway_Exit_192x256.avi")
        self.video_capture = cv2.VideoCapture(video_filepath)

        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

    def save_resized_video(self, target_size):
        height, width = target_size
        video_filepath = os.path.join(self.dataset_path, "Subway_Exit_{0}x{1}.avi".format(height, width))
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(video_filepath, self.fourcc, self.fps, frame_size)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            print("\r{0}/{1}".format(i + 1, self.frame_count), end='')
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, dsize=frame_size)
            video_writer.write(frame)

    def save_to_npy_shards(self, shard_size, start=0, end=-1, skip=0, subset=None):
        start = start if start >= 0 else self.frame_count + start + 1
        end = end if end >= 0 else self.frame_count + end + 1
        assert start < end <= self.frame_count
        subset = "{0:05d}-{1:05d}".format(start, end) if subset is None else subset
        frame_count = (end - start) // (skip + 1)
        shard_count = int(np.ceil(frame_count / shard_size))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        tqdm_desc = "Creating shard (size={0}, subset={1}, resolution={2}x{3})".format(
            shard_size, subset, self.frame_height, self.frame_width)
        for i in tqdm(range(shard_count), desc=tqdm_desc):
            if i == (shard_count - 1):
                shard_size = frame_count - shard_size * i
            shard = np.empty(shape=[shard_size, self.frame_height, self.frame_width])
            for j in range(shard_size):
                ret, frame = self.video_capture.read()
                shard[j] = np.mean(frame, axis=-1)
                if skip > 0:
                    next_index = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) + skip
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(next_index))
            shard_filename = "Subway_Exit_{0}x{1}.{2}.{3:03d}.npy".format(
                self.frame_height, self.frame_width, subset, i)
            shard_filename = os.path.join(self.dataset_path, shard_filename)
            shard = shard / 255.0
            np.save(shard_filename, shard)

    def make_copy(self, copy_inputs=False, copy_labels=False):
        pass

    def shuffle(self):
        pass

    def resized(self, size):
        pass

    def sample(self, batch_size=None, apply_preprocess_step=True, seed=None):
        pass

    def current_batch(self, batch_size: int = None, apply_preprocess_step=True):
        pass

    @property
    def samples_count(self):
        return self.frame_count

    @property
    def images_size(self):
        return self.frame_height, self.frame_width

    @property
    def frame_level_labels(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


subway_dataset = SubwayDataset(dataset_path=r"C:\Users\Degva\Downloads")
train_samples = 25 * 5 * 60
subway_dataset.save_to_npy_shards(shard_size=15000, start=0, end=train_samples, skip=0, subset="train")
subway_dataset.save_to_npy_shards(shard_size=15000, start=train_samples, end=-1, skip=4, subset="test")
# for n in range(6):
#     subway_dataset.save_resized_video(target_size=[192 // (2 ** n), 256 // (2 ** n)])
