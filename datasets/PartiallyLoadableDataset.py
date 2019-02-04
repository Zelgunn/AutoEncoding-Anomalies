import numpy as np
import copy
import random
import os
from typing import List

from datasets import Dataset
from data_preprocessors import DataPreprocessor


class PartiallyLoadableDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 config: dict,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        super(PartiallyLoadableDataset, self).__init__(data_preprocessors=data_preprocessors,
                                                       batch_size=batch_size,
                                                       epoch_length=epoch_length,
                                                       shuffle_on_epoch_end=shuffle_on_epoch_end,
                                                       **kwargs)
        self.dataset_path = dataset_path
        self.config = copy.deepcopy(config)
        self.sub_config_index = 0

    def shuffle(self):
        random.shuffle(self.config[self.sub_config_index]["filenames"])

    def resized(self, size):
        target_height, target_width = size
        target_index = None
        for i in range(len(self.config)):
            height, width = self.config[i]["images_size"]
            if (target_width == width) and (target_height == height):
                target_index = i
        assert target_index is not None, \
            "size {0}x{1} not in database".format(target_height, target_width)

        dataset_type = type(self)
        other = dataset_type(dataset_path=self.dataset_path, config=self.config,
                             data_preprocessors=self.data_preprocessors, batch_size=self.batch_size,
                             epoch_length=self.epoch_length, shuffle_on_epoch_end=self.shuffle_on_epoch_end)
        other.sub_config_index = target_index
        return other

    def sample(self, batch_size=None, apply_preprocess_step=True, seed=None):
        np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size

        shard_filename = self.filenames[np.random.randint(len(self.filenames))]
        shard_filepath = os.path.join(self.dataset_path, shard_filename)
        shard = np.load(shard_filepath, mmap_mode="r")
        indices = np.random.permutation(np.arange(len(shard)))[:batch_size]
        images = shard[indices]

        if apply_preprocess_step and (len(self.data_preprocessors) > 0):
            inputs, outputs = self.apply_preprocess(images, np.copy(images))
        else:
            inputs, outputs = images, images

        np.random.seed(None)
        return inputs, outputs

    def current_batch(self, batch_size: int = None, apply_preprocess_step=True):
        raise NotImplementedError

    @property
    def samples_count(self):
        return self.config[self.sub_config_index]["samples_count"]

    @property
    def images_size(self):
        return self.config[self.sub_config_index]["images_size"]

    @property
    def filenames(self):
        return self.config[self.sub_config_index]["filenames"]

    @property
    def frame_level_labels(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
