from keras.utils import Sequence
from abc import ABC, abstractmethod
import numpy as np
from typing import List

from data_preprocessors import DataPreprocessor
from utils.numpy_utils import NumpySeedContext


class Dataset(Sequence, ABC):
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 targets_are_predictions: bool,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.input_sequence_length = input_sequence_length

        self.output_sequence_length = output_sequence_length
        self.targets_are_predictions = targets_are_predictions
        self.data_preprocessors = data_preprocessors or []
        self.batch_size = batch_size
        self.epoch_length = epoch_length

        self.shuffle_on_epoch_end = shuffle_on_epoch_end

        self.saved_to_npz: bool = False
        self.index = 0
        self.epochs_completed = 0

        self._normalization_range = [None, None]

    def __len__(self):
        if self.epoch_length is None:
            return int(np.ceil(self.samples_count / self.batch_size))
        else:
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

    def get_batch(self, batch_size=None, seed=None, apply_preprocess_step=True, max_shard_count=1):
        with NumpySeedContext(seed):
            images: np.ndarray = self.sample(batch_size, max_shard_count=max_shard_count, seed=seed)

            inputs, outputs = self.divide_batch_io(images)

            if apply_preprocess_step and (len(self.data_preprocessors) > 0):
                inputs, outputs = self.apply_preprocess(inputs, outputs)
        return inputs, outputs

    @abstractmethod
    def sample(self, batch_size=None, seed=None, sequence_length=None, max_shard_count=1, return_labels=False):
        raise NotImplementedError

    @property
    def total_sequence_length(self):
        in_length = 1 if self.input_sequence_length is None else self.input_sequence_length
        out_length = 1 if self.output_sequence_length is None else self.output_sequence_length
        if self.targets_are_predictions:
            return in_length + out_length
        else:
            return max(in_length, out_length)

    def sample_indices(self, batch_size, indices_range, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.total_sequence_length

        indices = np.arange(indices_range - sequence_length + 1)
        indices = np.random.permutation(indices)[:batch_size]
        indices = np.repeat(indices, sequence_length)
        indices = np.reshape(indices, [batch_size, sequence_length])
        indices = indices + np.arange(sequence_length)

        return indices

    def divide_batch_io(self, images):
        if self.input_sequence_length is None:
            inputs = images[:, 0]
        else:
            inputs = images[:, :self.input_sequence_length]

        if self.output_sequence_length is None:
            outputs = images[:, -1]
        else:
            outputs = images[:, -self.output_sequence_length:]

        if not self.targets_are_predictions:
            if self.input_sequence_length < self.output_sequence_length:
                inputs = np.copy(inputs)
            else:
                outputs = np.copy(outputs)
        return inputs, outputs

    def sample_input_images(self, batch_size=None, seed=None, max_shard_count=1):
        return self.sample(batch_size, seed, sequence_length=self.input_sequence_length,
                           max_shard_count=max_shard_count, return_labels=False)

    @property
    @abstractmethod
    def samples_count(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def images_size(self):
        raise NotImplementedError

    @property
    def has_pixel_level_anomaly_labels(self):
        return False
