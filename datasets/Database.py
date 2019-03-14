from abc import ABC, abstractmethod
from typing import List

from data_preprocessors import DataPreprocessor, RandomCropper
from datasets import Dataset


class Database(ABC):
    def __init__(self,
                 database_path: str,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        self.database_path = database_path
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.train_preprocessors = train_preprocessors or []
        self.test_preprocessors = test_preprocessors or []
        self._require_saving = False
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def normalize(self, target_min=0.0, target_max=1.0):
        raise NotImplementedError

    def on_epoch_end(self):
        self.train_dataset.on_epoch_end()
        self.test_dataset.on_epoch_end()

    @property
    def images_size(self):
        return self.train_dataset.images_size

    def resized(self, image_size, input_sequence_length, output_sequence_length):
        if self.images_size == image_size:
            self.input_sequence_length = input_sequence_length
            self.output_sequence_length = output_sequence_length
            return self

        database_type = type(self)
        database: Database = database_type(self.database_path, input_sequence_length, output_sequence_length)

        if cropper_in_preprocessors(self.train_preprocessors):
            database.train_dataset = self.train_dataset
        else:
            database.train_dataset = self.train_dataset.resized(image_size, input_sequence_length,
                                                                output_sequence_length)

        if cropper_in_preprocessors(self.test_preprocessors):
            database.test_dataset = self.test_dataset
        else:
            database.test_dataset = self.test_dataset.resized(image_size, input_sequence_length, output_sequence_length)

        database._require_saving = self._require_saving

        return database


def cropper_in_preprocessors(preprocessors):
    result = False
    for preprocessor in preprocessors:
        if isinstance(preprocessor, RandomCropper):
            result = True
            break
    return result
