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
        self._shard_indices_offset = None

    def sample(self, batch_size=None, apply_preprocess_step=True, seed=None):
        np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size

        shard_index = np.random.randint(len(self.images_filenames))
        shard_filepath = os.path.join(self.dataset_path, self.images_filenames[shard_index])
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

    def sample_with_anomaly_labels(self, batch_size=None, seed=None, max_shard_count=1):
        np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size
        shard_size = batch_size // max_shard_count

        selected_shards = np.random.randint(len(self.images_filenames), size=max_shard_count)
        images = np.empty(shape=[batch_size, *self.images_size, 1], dtype=np.float32)
        labels = np.empty(shape=[batch_size], dtype=np.bool)

        batch_index = 0
        for i, shard_index in enumerate(selected_shards):
            images_shard_filepath = os.path.join(self.dataset_path, self.images_filenames[shard_index])
            labels_shard_filepath = os.path.join(self.dataset_path, self.labels_filenames[shard_index])

            images_shard = np.load(images_shard_filepath, mmap_mode="r")
            labels_shard = np.load(labels_shard_filepath, mmap_mode="r")

            if i == (len(selected_shards) - 1):
                shard_size += batch_size % max_shard_count

            shard_indices = np.random.permutation(np.arange(len(images_shard)))[:shard_size]
            images[batch_index:batch_index + shard_size] = images_shard[shard_indices]
            labels[batch_index:batch_index + shard_size] = labels_shard[shard_indices]

            batch_index += shard_size

        np.random.seed(None)

        return images, labels

    def shuffle(self):
        pass

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

    # region Properties
    @property
    def sub_config(self):
        return self.config[self.sub_config_index]

    @property
    def samples_count(self):
        return self.sub_config["samples_count"]

    @property
    def images_size(self):
        return self.sub_config["images_size"]

    @property
    def images_filenames(self):
        return self.sub_config["images_filenames"]

    @property
    def labels_filenames(self):
        return self.sub_config["labels_filenames"]
    # endregion
