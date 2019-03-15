from datasets.ucsd.UCSDSubset import UCSDSubset
import os
from typing import List

from datasets import FullyLoadableDatabase
from data_preprocessors import DataPreprocessor


class UCSDDatabase(FullyLoadableDatabase):
    def load(self):
        self.train_subset = self.load_subset("Train", self.train_preprocessors)
        self.test_subset = self.load_subset("Test", self.test_preprocessors)

        if not self.train_subset.saved_to_npz:
            self.normalize()
            self.train_subset.save_to_npz()
            self.test_subset.save_to_npz()

    def load_subset(self, subset_name: str, data_preprocessors: List[DataPreprocessor]):
        subset_path = os.path.join(self.database_path, subset_name)
        subset = UCSDSubset(subset_path=subset_path, data_preprocessors=data_preprocessors,
                            input_sequence_length=self.input_sequence_length,
                            output_sequence_length=self.output_sequence_length,
                            epoch_length=-1)
        subset.load()
        return subset
