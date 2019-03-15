from abc import ABC
import numpy as np
import cv2
from typing import List

from datasets import Dataset, FullyLoadableSubset
from data_preprocessors import DataPreprocessor


class FullyLoadableDataset(Dataset, ABC):
    def __init__(self,
                 dataset_path: str,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        super(FullyLoadableDataset, self).__init__(dataset_path=dataset_path,
                                                   input_sequence_length=input_sequence_length,
                                                   output_sequence_length=output_sequence_length,
                                                   train_preprocessors=train_preprocessors,
                                                   test_preprocessors=test_preprocessors)
        self.train_subset: FullyLoadableSubset = self.train_subset
        self.test_subset: FullyLoadableSubset = self.test_subset

    def normalize(self, target_min=0.0, target_max=1.0):
        current_min = None
        current_max = None
        for i in range(self.train_subset.videos_count):
            video_min = self.train_subset.videos[i].min()
            video_max = self.train_subset.videos[i].max()
            if current_min is None:
                current_min = video_min
                current_max = video_max
            else:
                current_min = min(current_min, video_min)
                current_max = max(current_max, video_max)

        for i in range(self.test_subset.videos_count):
            video_min = self.test_subset.videos[i].min()
            video_max = self.test_subset.videos[i].max()

            current_min = min(current_min, video_min)
            current_max = max(current_max, video_max)

        self.train_subset.normalize(current_min, current_max, target_min, target_max)
        self.test_subset.normalize(current_min, current_max, target_min, target_max)

    def visualize_test_subset(self):
        pixel_labels = self.test_subset.pixel_labels

        for i in range(len(pixel_labels)):
            tmp: np.ndarray = self.test_subset.pixel_labels[i]
            tmp = tmp.astype(np.float32)
            inv_tmp = 1.0 - tmp
            tmp = self.test_subset.videos[i] * tmp
            tmp += self.test_subset.videos[i] * inv_tmp * 0.25
            cv2.imshow("test_subset", tmp)
            cv2.waitKey(30)
