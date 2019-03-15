from abc import ABC, abstractmethod
from typing import List

from data_preprocessors import DataPreprocessor, RandomCropper
from datasets import Subset


class Dataset(ABC):
    def __init__(self,
                 dataset_path: str,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        self.dataset_path = dataset_path
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.train_preprocessors = train_preprocessors or []
        self.test_preprocessors = test_preprocessors or []
        self._require_saving = False
        self.train_subset: Subset = None
        self.test_subset: Subset = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def normalize(self, target_min=0.0, target_max=1.0):
        raise NotImplementedError

    def on_epoch_end(self):
        self.train_subset.on_epoch_end()
        self.test_subset.on_epoch_end()

    @property
    def images_size(self):
        return self.train_subset.images_size

    def resized(self, image_size, input_sequence_length, output_sequence_length):
        if self.images_size == image_size:
            self.input_sequence_length = input_sequence_length
            self.output_sequence_length = output_sequence_length
            return self

        dataset_type = type(self)
        dataset: Dataset = dataset_type(self.dataset_path, input_sequence_length, output_sequence_length)

        if cropper_in_preprocessors(self.train_preprocessors):
            dataset.train_subset = self.train_subset
        else:
            dataset.train_subset = self.train_subset.resized(image_size, input_sequence_length,
                                                             output_sequence_length)

        if cropper_in_preprocessors(self.test_preprocessors):
            dataset.test_subset = self.test_subset
        else:
            dataset.test_subset = self.test_subset.resized(image_size, input_sequence_length, output_sequence_length)

        dataset._require_saving = self._require_saving

        return dataset


def cropper_in_preprocessors(preprocessors):
    result = False
    for preprocessor in preprocessors:
        if isinstance(preprocessor, RandomCropper):
            result = True
            break
    return result
