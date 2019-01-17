from abc import ABC, abstractmethod
import sys

from scheme import Dataset


class Database(ABC):
    def __init__(self, **kwargs):
        self._require_saving = False
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None

        if "database_path" in kwargs:
            self.database_path: str = kwargs["database_path"]
            self.load(self.database_path)
        else:
            self.database_path: str = None

    @abstractmethod
    def load(self, database_path, **kwargs):
        raise NotImplementedError

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
