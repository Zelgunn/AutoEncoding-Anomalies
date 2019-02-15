from datasets.ucsd.UCSDDataset import UCSDDataset
import os
from typing import List

from datasets import FullyLoadableDatabase
from data_preprocessors import DataPreprocessor


class UCSDDatabase(FullyLoadableDatabase):
    def load(self):
        self.train_dataset = self.load_dataset("Train", self.train_preprocessors)
        self.test_dataset = self.load_dataset("Test", self.test_preprocessors)

        if not self.train_dataset.saved_to_npz:
            self.normalize()
            self.train_dataset.save_to_npz()
            self.test_dataset.save_to_npz()

    def load_dataset(self, dataset_name: str, data_preprocessors: List[DataPreprocessor]):
        dataset_path = os.path.join(self.database_path, dataset_name)
        dataset = UCSDDataset(dataset_path=dataset_path, data_preprocessors=data_preprocessors,
                              input_sequence_length=self.input_sequence_length,
                              output_sequence_length=self.output_sequence_length,
                              targets_are_predictions=self.targets_are_predictions)
        dataset.load()
        return dataset
