from typing import Dict

from datasets.loaders import SubsetLoader, DatasetConfig


class DatasetLoader(object):
    def __init__(self, dataset_path: str, config: DatasetConfig):
        self.dataset_path = dataset_path
        self.config = config

        self.subsets: Dict[str, SubsetLoader] = {}
        for subset_name in config.subsets:
            self.subsets[subset_name] = SubsetLoader(config, subset_name)

    @property
    def train_subset(self) -> SubsetLoader:
        return self.subsets["Train"]

    @property
    def test_subset(self) -> SubsetLoader:
        return self.subsets["Test"]
