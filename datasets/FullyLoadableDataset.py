from abc import ABC
import numpy as np
import cv2
import copy
from typing import List

from datasets import Dataset
from data_preprocessors import DataPreprocessor


class FullyLoadableDataset(Dataset, ABC):
    # region Initialization
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 dataset_path: str,
                 epoch_length: int,
                 batch_size=64,
                 data_preprocessors: List[DataPreprocessor] = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.dataset_path = dataset_path
        self.videos = None
        self.anomaly_labels = None
        self._frame_level_labels = None
        super(FullyLoadableDataset, self).__init__(input_sequence_length=input_sequence_length,
                                                   output_sequence_length=output_sequence_length,
                                                   data_preprocessors=data_preprocessors,
                                                   batch_size=batch_size,
                                                   epoch_length=epoch_length,
                                                   shuffle_on_epoch_end=shuffle_on_epoch_end,
                                                   **kwargs)

    def make_copy(self, copy_inputs=False, copy_labels=False):
        dataset_type = type(self)
        other: FullyLoadableDataset = dataset_type(dataset_path=self.dataset_path,
                                                   input_sequence_length=self.input_sequence_length,
                                                   output_sequence_length=self.output_sequence_length,
                                                   epoch_length=self.epoch_length)
        other.data_preprocessors = self.data_preprocessors
        other.batch_size = self.batch_size
        other.shuffle_on_epoch_end = self.shuffle_on_epoch_end

        other.saved_to_npz = self.saved_to_npz
        other.index = self.index
        other.epochs_completed = self.epochs_completed
        other._normalization_range = copy.copy(self._normalization_range)

        other.videos = np.copy(self.videos) if copy_inputs else self.videos
        if self.anomaly_labels is not None:
            if copy_labels:
                other.anomaly_labels = np.copy(self.anomaly_labels)
                other._frame_level_labels = np.copy(self.frame_level_labels)
            else:
                other.anomaly_labels = self.anomaly_labels
                other._frame_level_labels = self.frame_level_labels
        return other

    # endregion

    def normalize(self, current_min, current_max, target_min=0.0, target_max=1.0):
        super(FullyLoadableDataset, self).normalize(current_min, current_max, target_min, target_max)
        multiplier = (target_max - target_min) / (current_max - current_min)
        for i in range(self.videos_count):
            self.videos[i] = (self.videos[i] - current_min) * multiplier + target_min

    def sample_single_video(self, video_index, shard_size, sequence_length, return_labels):
        video = self.videos[video_index]

        indices = self.sample_indices(shard_size, len(video), sequence_length)
        video = video[indices]

        frame_labels = pixel_labels = None
        if return_labels:
            labels_indices = indices[:, -self.output_sequence_length:]
            labels = self.anomaly_labels[video_index][labels_indices]

            if self.has_pixel_level_anomaly_labels:
                pixel_labels = labels
                frame_labels = self.frame_level_labels[video_index][labels_indices]
            else:
                frame_labels = labels

        return video, frame_labels, pixel_labels

    def get_video_length(self, video_index):
        return len(self.videos[video_index])

    def get_video_frames(self, video_index, start, end):
        return self.videos[video_index][start:end]

    def get_video_frame_labels(self, video_index, start, end):
        labels_array = self.frame_level_labels if self.has_pixel_level_anomaly_labels else self.anomaly_labels
        return labels_array[video_index][start:end]

    def shuffle(self):
        if self.anomaly_labels is None:
            np.random.shuffle(self.videos)
        else:
            shuffle_indices = np.random.permutation(np.arange(self.videos_count))
            self.videos = self.videos[shuffle_indices]
            self.anomaly_labels = self.anomaly_labels[shuffle_indices]

    # region Resizing
    def resized(self, size, input_sequence_length, output_sequence_length):
        videos = self.resized_videos(size)
        anomaly_labels = self.resized_anomaly_labels(size)

        dataset = self.make_copy()
        dataset.videos = videos
        dataset.input_sequence_length = input_sequence_length
        dataset.output_sequence_length = output_sequence_length

        if dataset.anomaly_labels is not None:
            dataset.anomaly_labels = anomaly_labels
            dataset._frame_level_labels = np.copy(self.frame_level_labels)
        return dataset

    def resized_videos(self, size):
        return resize_videos(self.videos, size)

    def resized_anomaly_labels(self, size):
        if self.anomaly_labels is None:
            anomaly_labels = None
        else:
            anomaly_labels = []
            for i in range(len(self.anomaly_labels)):
                video_labels = self.anomaly_labels[i]
                video_labels = video_labels.astype(np.float32)
                video_labels = resize_images(video_labels, size)
                video_labels = video_labels.astype(np.bool)
                anomaly_labels.append(video_labels)
            anomaly_labels = np.array(anomaly_labels)

        return anomaly_labels

    # endregion

    # region Properties
    @property
    def videos_count(self):
        return len(self.videos)

    def samples_count(self) -> int:
        count = 0
        for i in range(self.videos_count):
            count += len(self.videos[i])
        return count

    @property
    def images_size(self):
        return self.videos[0].shape[1:3]

    @property
    def has_labels(self):
        return self.anomaly_labels is not None

    @property
    def frame_level_labels(self):
        assert self.has_labels

        if not self.has_pixel_level_anomaly_labels:
            return self.anomaly_labels

        if self._frame_level_labels is None:
            self._frame_level_labels = []
            for i in range(len(self.anomaly_labels)):
                pixel_labels: np.ndarray = self.anomaly_labels[i]
                frame_labels = np.any(pixel_labels, axis=(1, 2, 3))
                self._frame_level_labels.append(frame_labels)
            self._frame_level_labels = np.array(self.frame_level_labels)

        return self._frame_level_labels
    # endregion


def resize_videos(videos, size) -> np.ndarray:
    videos_count = len(videos)
    resized_videos = []
    for i in range(videos_count):
        resized_video = resize_images(videos[i], size)
        resized_videos.append(resized_video)

    resized_videos = np.array(resized_videos)
    return resized_videos


def resize_images(images, size) -> np.ndarray:
    result = np.empty((images.shape[0], *size, images.shape[3]))
    dsize = tuple(reversed(size))
    for i in range(images.shape[0]):
        resized_one = cv2.resize(images[i], dsize, interpolation=cv2.INTER_AREA)
        if images.shape[3] == 1:
            result[i, :, :, 0] = resized_one
        else:
            result[i, :, :] = resized_one
    return result
