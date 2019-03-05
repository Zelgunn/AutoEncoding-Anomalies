from keras.utils import Sequence
from abc import ABC, abstractmethod
import numpy as np
from typing import List

from data_preprocessors import DataPreprocessor
from utils.numpy_utils import NumpySeedContext, fast_concatenate_0


class Dataset(Sequence, ABC):
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 epoch_length: int,
                 batch_size=64,
                 data_preprocessors: List[DataPreprocessor] = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.input_sequence_length = input_sequence_length

        self.output_sequence_length = output_sequence_length
        self.data_preprocessors = data_preprocessors or []
        self.batch_size = batch_size
        self.epoch_length = epoch_length

        self.shuffle_on_epoch_end = shuffle_on_epoch_end

        self.saved_to_npz: bool = False
        self.index = 0
        self.epochs_completed = 0

        self._normalization_range = [None, None]

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index):
        if self.epoch_length is None:
            raise NotImplementedError
        else:
            batch = self.get_batch()
        self.index += 1
        return batch

    def normalize(self, current_min, current_max, target_min=0.0, target_max=1.0):
        self._normalization_range = [target_min, target_max]

    def apply_preprocess(self, inputs, outputs):
        inputs = np.copy(inputs)
        outputs = np.copy(outputs)
        for data_preprocessor in self.data_preprocessors:
            inputs, outputs = data_preprocessor.process(inputs, outputs)
        return inputs, outputs

    def on_epoch_end(self):
        pass

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError

    @abstractmethod
    def resized(self, size, input_sequence_length, output_sequence_length):
        raise NotImplementedError

    def get_batch(self, batch_size=None, seed=None, apply_preprocess_step=True, max_shard_count=8):
        with NumpySeedContext(seed):
            videos: np.ndarray = self.sample(batch_size, seed=seed, sampled_videos_count=max_shard_count)

            inputs, outputs = self.divide_batch_io(videos)

            if apply_preprocess_step and (len(self.data_preprocessors) > 0):
                inputs, outputs = self.apply_preprocess(inputs, outputs)
        return inputs, outputs

    def sample(self, batch_size=None, seed=None, sequence_length=None, sampled_videos_count=None, return_labels=False):
        if batch_size is None:
            batch_size = self.batch_size

        if sampled_videos_count is None:
            sampled_videos_count = self.videos_count
        else:
            sampled_videos_count = min(sampled_videos_count, self.videos_count)

        if batch_size < sampled_videos_count:
            sampled_videos_count = batch_size

        with NumpySeedContext(seed):
            videos_indices = np.random.permutation(np.arange(self.videos_count))[:sampled_videos_count]
            sampled_videos, sampled_frame_labels, sampled_pixel_labels,  = [], [], None
            shard_size = batch_size // sampled_videos_count

            for i, video_index in enumerate(videos_indices):
                if i == (sampled_videos_count - 1):
                    shard_size += batch_size % sampled_videos_count

                video, frame_labels, pixel_labels = self.sample_single_video(video_index, shard_size, sequence_length,
                                                                             return_labels)
                sampled_videos.append(video)

                if return_labels:
                    sampled_frame_labels.append(frame_labels)
                    if pixel_labels is not None:
                        if sampled_pixel_labels is None:
                            sampled_pixel_labels = []
                        sampled_pixel_labels.append(pixel_labels)

        sampled_videos = fast_concatenate_0(sampled_videos)

        if return_labels:
            sampled_frame_labels = fast_concatenate_0(sampled_frame_labels)
            if sampled_pixel_labels is not None:
                sampled_pixel_labels = fast_concatenate_0(sampled_pixel_labels)
            return sampled_videos, sampled_frame_labels, sampled_pixel_labels
        else:
            return sampled_videos

    @abstractmethod
    def sample_single_video(self, video_index, shard_size, sequence_length, return_labels):
        raise NotImplementedError

    @abstractmethod
    def get_video_length(self, video_index):
        raise NotImplementedError

    @abstractmethod
    def get_video_frames(self, video_index, start, end):
        raise NotImplementedError

    @abstractmethod
    def get_video_frame_labels(self, video_index, start, end):
        raise NotImplementedError

    @property
    def total_sequence_length(self):
        return max(self.input_sequence_length, self.output_sequence_length)

    def sample_indices(self, batch_size, indices_range, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.total_sequence_length

        indices = np.arange(indices_range - sequence_length + 1)
        indices = np.random.permutation(indices)[:batch_size]
        indices = np.repeat(indices, sequence_length)
        indices = np.reshape(indices, [batch_size, sequence_length])
        indices = indices + np.arange(sequence_length)

        return indices

    def divide_batch_io(self, videos):
        inputs = videos[:, :self.input_sequence_length]
        outputs = videos[:, -self.output_sequence_length:]
        return inputs, outputs

    def sample_input_videos(self, batch_size=None, seed=None, max_shard_count=1):
        return self.sample(batch_size, seed, sequence_length=self.input_sequence_length,
                           sampled_videos_count=max_shard_count, return_labels=False)

    @property
    @abstractmethod
    def videos_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def samples_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def images_size(self):
        raise NotImplementedError

    @property
    def has_pixel_level_anomaly_labels(self):
        return False
