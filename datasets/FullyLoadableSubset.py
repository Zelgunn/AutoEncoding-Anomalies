from abc import ABC
import numpy as np
import cv2
import copy
from typing import List

from datasets import Subset
from data_preprocessors import DataPreprocessor


class FullyLoadableSubset(Subset, ABC):
    # region Initialization
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 subset_path: str,
                 epoch_length: int,
                 batch_size=64,
                 data_preprocessors: List[DataPreprocessor] = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.subset_path = subset_path
        self.videos = None
        self.pixel_labels = None
        self._frame_labels = None
        super(FullyLoadableSubset, self).__init__(input_sequence_length=input_sequence_length,
                                                  output_sequence_length=output_sequence_length,
                                                  data_preprocessors=data_preprocessors,
                                                  batch_size=batch_size,
                                                  epoch_length=epoch_length,
                                                  shuffle_on_epoch_end=shuffle_on_epoch_end,
                                                  **kwargs)

    def make_copy(self, copy_inputs=False, copy_labels=False):
        subset_type = type(self)
        other: FullyLoadableSubset = subset_type(subset_path=self.subset_path,
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
        if copy_labels:
            if self.pixel_labels is not None:
                other.pixel_labels = np.copy(self.pixel_labels)
            other._frame_labels = np.copy(self.frame_labels)
        else:
            other.pixel_labels = self.pixel_labels
            other._frame_labels = self.frame_labels
        return other

    # endregion

    def normalize(self, current_min, current_max, target_min=0.0, target_max=1.0):
        super(FullyLoadableSubset, self).normalize(current_min, current_max, target_min, target_max)
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

            frame_labels = self.frame_labels[video_index][labels_indices]
            if self.has_pixel_labels:
                pixel_labels = self.pixel_labels[video_index][labels_indices]

        return video, frame_labels, pixel_labels

    def get_video_length(self, video_index):
        return len(self.videos[video_index])

    def get_video_frames(self, video_index, start, end):
        return self.videos[video_index][start:end]

    def get_video_frame_labels(self, video_index, start, end):
        return self.frame_labels[video_index][start:end]

    def get_video_pixel_labels(self, video_index, start, end):
        return self.pixel_labels[video_index][start:end]

    # region Resizing
    def resized(self, size, input_sequence_length, output_sequence_length):
        subset = self.make_copy(copy_inputs=False, copy_labels=False)
        subset.videos = self.resized_videos(size)
        subset.input_sequence_length = input_sequence_length
        subset.output_sequence_length = output_sequence_length

        if self._frame_labels is not None:
            subset._frame_labels = np.copy(self._frame_labels)

        if self.pixel_labels is not None:
            subset.pixel_labels = self.resized_pixel_labels(size)

        return subset

    def resized_videos(self, size):
        return resize_videos(self.videos, size)

    def resized_pixel_labels(self, size):
        if self.pixel_labels is None:
            pixel_labels = None
        else:
            pixel_labels = []
            for i in range(len(self.pixel_labels)):
                video_labels = self.pixel_labels[i]
                video_labels = video_labels.astype(np.float32)
                video_labels = resize_images(video_labels, size)
                video_labels = video_labels.astype(np.bool)
                pixel_labels.append(video_labels)
            pixel_labels = np.array(pixel_labels)

        return pixel_labels

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
        return (self.pixel_labels is not None) or (self._frame_labels is not None)

    @property
    def frame_labels(self):
        if not self.has_labels:
            return None

        if self._frame_labels is None:
            self._frame_labels = []
            for i in range(len(self.pixel_labels)):
                pixel_labels: np.ndarray = self.pixel_labels[i]
                frame_labels = np.any(pixel_labels, axis=(1, 2, 3))
                self._frame_labels.append(frame_labels)
            self._frame_labels = np.array(self.frame_labels)

        return self._frame_labels
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
