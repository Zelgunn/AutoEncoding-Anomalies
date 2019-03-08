from abc import ABC
import numpy as np
import cv2
from typing import List

from datasets import Database, FullyLoadableDataset
from data_preprocessors import DataPreprocessor


class FullyLoadableDatabase(Database, ABC):
    def __init__(self,
                 database_path: str,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        super(FullyLoadableDatabase, self).__init__(database_path=database_path,
                                                    input_sequence_length=input_sequence_length,
                                                    output_sequence_length=output_sequence_length,
                                                    train_preprocessors=train_preprocessors,
                                                    test_preprocessors=test_preprocessors)
        self.train_dataset: FullyLoadableDataset = self.train_dataset
        self.test_dataset: FullyLoadableDataset = self.test_dataset

    def normalize(self, target_min=0.0, target_max=1.0):
        current_min = None
        current_max = None
        for i in range(self.train_dataset.videos_count):
            video_min = self.train_dataset.videos[i].min()
            video_max = self.train_dataset.videos[i].max()
            if current_min is None:
                current_min = video_min
                current_max = video_max
            else:
                current_min = min(current_min, video_min)
                current_max = max(current_max, video_max)

        for i in range(self.test_dataset.videos_count):
            video_min = self.test_dataset.videos[i].min()
            video_max = self.test_dataset.videos[i].max()

            current_min = min(current_min, video_min)
            current_max = max(current_max, video_max)

        self.train_dataset.normalize(current_min, current_max, target_min, target_max)
        self.test_dataset.normalize(current_min, current_max, target_min, target_max)

    def visualize_test_dataset(self):
        pixel_labels = self.test_dataset.pixel_labels

        for i in range(len(pixel_labels)):
            tmp: np.ndarray = self.test_dataset.pixel_labels[i]
            tmp = tmp.astype(np.float32)
            inv_tmp = 1.0 - tmp
            tmp = self.test_dataset.videos[i] * tmp
            tmp += self.test_dataset.videos[i] * inv_tmp * 0.25
            cv2.imshow("test_dataset", tmp)
            cv2.waitKey(30)
