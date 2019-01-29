from keras.utils import Sequence
from abc import ABC, abstractmethod
import copy
from typing import List

from data_preprocessors import DataPreprocessor


class Dataset(Sequence, ABC):
    def __init__(self,
                 dataset_path: str = None,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        self.data_preprocessors = data_preprocessors or []
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.shuffle_on_epoch_end = shuffle_on_epoch_end

        self.dataset_path: str = None
        self.saved_to_npz: bool = False
        self.index = 0
        self.epochs_completed = 0

        self._normalization_ranges = [None, None, None, None]

        if dataset_path is not None:
            self.load(dataset_path)

    @abstractmethod
    def load(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path

    @abstractmethod
    def make_copy(self, copy_inputs=False, copy_labels=False):
        dataset_type = type(self)
        other: Dataset = dataset_type()
        other.data_preprocessors = self.data_preprocessors
        other.batch_size = self.batch_size
        other.epoch_length = self.epoch_length
        other.shuffle_on_epoch_end = self.shuffle_on_epoch_end
        other.dataset_path = self.dataset_path
        other.saved_to_npz = self.saved_to_npz
        other.index = self.index
        other.epochs_completed = self.epochs_completed
        other._normalization_ranges = copy.copy(self._normalization_ranges)
        return other

    def set_normalization_ranges(self, current_min, current_max, target_min=0.0, target_max=1.0):
        self._normalization_ranges = [current_min, current_max, target_min, target_max]

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

    @property
    @abstractmethod
    def samples_count(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def images_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_level_labels(self):
        raise NotImplementedError
