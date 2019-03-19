from typing import Dict

from datasets import SubsetV2, DatasetConfigV2


class DatasetV2(object):
    def __init__(self, dataset_path: str, config: DatasetConfigV2):
        self.dataset_path = dataset_path
        self.config = config

        self.subsets: Dict[str, SubsetV2] = None

    @property
    def train_subset(self) -> SubsetV2:
        return self.subsets["train"]

    @property
    def test_subset(self) -> SubsetV2:
        return self.subsets["test"]
