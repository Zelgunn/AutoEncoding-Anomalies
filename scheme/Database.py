from abc import ABC, abstractmethod
import sys
import numpy as np
import cv2

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

    def normalize(self, target_min=0.0, target_max=1.0):
        current_min = min(self.train_dataset.images.min(), self.test_dataset.images.min())
        current_max = max(self.train_dataset.images.max(), self.test_dataset.images.max())
        self.train_dataset.normalize(current_min, current_max, target_min, target_max)
        self.test_dataset.normalize(current_min, current_max, target_min, target_max)

    def shuffle(self, seed=None):
        np.random.seed(seed)
        self.train_dataset.shuffle()
        self.test_dataset.shuffle()

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

    def visualize_test_dataset(self):
        labels = self.test_dataset.anomaly_labels

        for i in range(len(labels)):
            tmp: np.ndarray = self.test_dataset.anomaly_labels[i]
            tmp = tmp.astype(np.float32)
            inv_tmp = 1.0 - tmp
            tmp = self.test_dataset.images[i] * tmp
            tmp += self.test_dataset.images[i] * inv_tmp * 0.25
            cv2.imshow("test_dataset", tmp)
            cv2.waitKey(30)
