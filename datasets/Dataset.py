from keras.utils import Sequence
from abc import ABC, abstractmethod
import numpy as np
from typing import List

from data_preprocessors import DataPreprocessor


class Dataset(Sequence, ABC):
    def __init__(self,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
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
        batch = self.current_batch() if self.epoch_length is None else self.sample()
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
    def resized(self, size):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size=None, apply_preprocess_step=True, seed=None):
        raise NotImplementedError

    @abstractmethod
    def current_batch(self, batch_size: int = None, apply_preprocess_step=True):
        raise NotImplementedError

    @abstractmethod
    def sample_with_anomaly_labels(self, batch_size=None, seed=None, max_shard_count=1):
        raise NotImplementedError

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
