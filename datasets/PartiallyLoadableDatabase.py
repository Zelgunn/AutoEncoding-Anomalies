import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
import json
from typing import List

from datasets import Database, PartiallyLoadableDataset
from data_preprocessors import DataPreprocessor


class PartiallyLoadableDatabase(Database, ABC):
    def __init__(self,
                 database_path: str,
                 train_preprocessors: List[DataPreprocessor] = None,
                 test_preprocessors: List[DataPreprocessor] = None):
        super(PartiallyLoadableDatabase, self).__init__(database_path=database_path,
                                                        train_preprocessors=train_preprocessors,
                                                        test_preprocessors=test_preprocessors)
        self.header = None
        self.train_dataset: PartiallyLoadableDataset = self.train_dataset
        self.test_dataset: PartiallyLoadableDataset = self.test_dataset

    def load(self):
        with open(self.header_filepath, "r") as header_file:
            self.header = json.load(header_file)
        self.train_dataset = PartiallyLoadableDataset(dataset_path=self.database_path,
                                                      config=self.header["train"],
                                                      data_preprocessors=self.train_preprocessors)

        self.test_dataset = PartiallyLoadableDataset(dataset_path=self.database_path,
                                                     config=self.header["test"],
                                                     data_preprocessors=self.test_preprocessors)

    def prepare_resolutions(self, resolutions: List, shard_size=5000, skip=0):
        header = {"train": [None] * len(resolutions), "test": [None] * len(resolutions)}

        for i, (height, width) in enumerate(resolutions):
            print("===== Preparing resolution {0}x{1} ({2}/{3}) =====".format(height, width, i + 1, len(resolutions)))
            self.on_build_shard_begin(height, width)

            shards_dict = self.build_shards(shard_size, skip)

            header["train"][i] = shards_dict["train"]
            header["test"][i] = shards_dict["test"]

            if "min" not in header:
                header["min"] = shards_dict["min"]
                header["max"] = shards_dict["max"]

        with open(self.header_filepath, "w") as header_file:
            json.dump(header, header_file)

        self.header = header

    @abstractmethod
    def on_build_shard_begin(self, width, height):
        raise NotImplementedError

    def build_shards(self, shard_size, skip=0):
        shards_dict = {}
        shards_min, shards_max = None, None

        for i, (dataset_id, shard) in enumerate(self.build_shards_iterator(shard_size, skip)):
            height, width = np.shape(shard)[1:3]
            shard_filename = "{0}_{1}_{2}x{3}_{4:03d}.npy".format(self.base_name, dataset_id, height, width, i)
            shard_filepath = os.path.join(self.database_path, shard_filename)
            np.save(shard_filepath, shard)

            if i == 0:
                shards_min, shards_max = np.min(shard), np.max(shard)
            else:
                shards_min = min(np.min(shard), shards_min)
                shards_max = max(np.max(shard), shards_max)

            shard_size = np.shape(shard)[0]
            if dataset_id in shards_dict:
                shards_dict[dataset_id]["filenames"] += [shard_filename]
                shards_dict[dataset_id]["samples_count"] += shard_size
            else:
                shards_dict[dataset_id] = {"filenames": [shard_filename],
                                           "images_size": np.shape(shard)[1:3],
                                           "samples_count": shard_size}

        shards_dict["min"] = shards_min
        shards_dict["max"] = shards_max

        return shards_dict

    @abstractmethod
    def build_shards_iterator(self, shard_size, skip=0):
        raise NotImplementedError

    @property
    @abstractmethod
    def base_name(self):
        raise NotImplementedError

    @property
    def header_filename(self):
        return "{0}_header.json".format(self.base_name)

    @property
    def header_filepath(self):
        return os.path.join(self.database_path, self.header_filename)

    def normalize(self, target_min=0.0, target_max=1.0):
        pass
