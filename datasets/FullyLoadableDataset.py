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
                 dataset_path: str,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.dataset_path = dataset_path
        self.images = None
        self.anomaly_labels = None
        self._frame_level_labels = None
        super(FullyLoadableDataset, self).__init__(data_preprocessors=data_preprocessors,
                                                   batch_size=batch_size,
                                                   epoch_length=epoch_length,
                                                   shuffle_on_epoch_end=shuffle_on_epoch_end,
                                                   **kwargs)

    def make_copy(self, copy_inputs=False, copy_labels=False):
        dataset_type = type(self)
        other: FullyLoadableDataset = dataset_type(dataset_path=self.dataset_path)
        other.data_preprocessors = self.data_preprocessors
        other.batch_size = self.batch_size
        other.epoch_length = self.epoch_length
        other.shuffle_on_epoch_end = self.shuffle_on_epoch_end

        other.saved_to_npz = self.saved_to_npz
        other.index = self.index
        other.epochs_completed = self.epochs_completed
        other._normalization_range = copy.copy(self._normalization_range)

        other.images = np.copy(self.images) if copy_inputs else self.images
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
        self.images = (self.images - current_min) * multiplier + target_min

    def sample(self, batch_size=None, apply_preprocess_step=True, seed=None):
        np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size

        indices = np.random.permutation(np.arange(self.images.shape[0]))[:batch_size]
        images: np.ndarray = self.images[indices]

        if apply_preprocess_step and (len(self.data_preprocessors) > 0):
            inputs, outputs = self.apply_preprocess(images, np.copy(images))
        else:
            inputs, outputs = images, images

        np.random.seed(None)
        return inputs, outputs

    def current_batch(self, batch_size: int = None, apply_preprocess_step=True):
        if batch_size is None:
            batch_size = self.batch_size
        images: np.ndarray = self.images[self.index * batch_size: (self.index + 1) * batch_size]

        if apply_preprocess_step and (len(self.data_preprocessors) > 0):
            inputs, outputs = self.apply_preprocess(images, np.copy(images))
        else:
            inputs, outputs = images, images

        return inputs, outputs

    def shuffle(self):
        if self.anomaly_labels is None:
            np.random.shuffle(self.images)
        else:
            shuffle_indices = np.random.permutation(np.arange(self.samples_count))
            self.images = self.images[shuffle_indices]
            self.anomaly_labels = self.anomaly_labels[shuffle_indices]

    # region Resizing
    def resized(self, size):
        images = self.resized_images(size)
        anomaly_labels = self.resized_anomaly_labels(size)

        dataset = self.make_copy()
        dataset.images = images

        if dataset.anomaly_labels is not None:
            dataset.anomaly_labels = anomaly_labels
            dataset._frame_level_labels = np.copy(self.frame_level_labels)
        return dataset

    def resized_images(self, size):
        return resize_images(self.images, size)

    def resized_anomaly_labels(self, size):
        if self.anomaly_labels is None:
            anomaly_labels = None
        else:
            anomaly_labels = resize_images(self.anomaly_labels.astype(np.float32), size).astype(bool)
        return anomaly_labels

    # endregion

    # region Properties
    @property
    def samples_count(self):
        return self.images.shape[0]

    @property
    def images_size(self):
        return self.images.shape[1:3]

    @property
    def frame_level_labels(self):
        assert self.anomaly_labels is not None
        if self._frame_level_labels is None:
            self._frame_level_labels = np.any(self.anomaly_labels, axis=(1, 2, 3))
        return self._frame_level_labels
    # endregion


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
