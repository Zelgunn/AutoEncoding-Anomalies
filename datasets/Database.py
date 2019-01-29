from abc import ABC, abstractmethod
import sys
import numpy as np
from typing import List

from data_preprocessors import DataPreprocessor
from datasets import Dataset


class Database(ABC):
    def __init__(self,
                 database_path: str = None,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        self.database_path = None
        self.train_preprocessors = train_preprocessors or []
        self.test_preprocessors = test_preprocessors or []
        self._require_saving = False
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None

        if database_path is not None:
            self.load(database_path)

    @abstractmethod
    def load(self, database_path):
        self.database_path = database_path

    @abstractmethod
    def normalize(self, target_min=0.0, target_max=1.0):
        raise NotImplementedError

    def on_epoch_end(self):
        self.train_dataset.on_epoch_end()
        self.test_dataset.on_epoch_end()

    def shuffle(self, seed=None):
        np.random.seed(seed)
        self.train_dataset.shuffle()
        self.test_dataset.shuffle()
        np.random.seed(None)

    @property
    def images_size(self):
        return self.train_dataset.images_size

    def resized_to_scale(self, scales_shape):
        scale_size = tuple(scales_shape[:2])
        if self.images_size == scale_size:
            return self

        database_type = type(self)
        database: Database = database_type()
        database.train_dataset = self.train_dataset.resized(scale_size)
        database.test_dataset = self.test_dataset.resized(scale_size)
        database.database_path = self.database_path
        database._require_saving = self._require_saving

        return database

    def resized_to_scales(self, scales_shapes):
        print("Generating images for different scales...", end='')
        sys.stdout.flush()

        databases = []

        for i in range(len(scales_shapes)):
            database = self.resized_to_scale(scales_shapes[i])
            databases.append(database)

        print(" Done !")
        return databases
