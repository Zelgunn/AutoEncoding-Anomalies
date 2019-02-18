import numpy as np
import copy
import os
from typing import List

from datasets import Dataset
from data_preprocessors import DataPreprocessor
from utils.numpy_utils import NumpySeedContext, fast_concatenate_0


class PartiallyLoadableDataset(Dataset):
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 targets_are_predictions: bool,
                 dataset_path: str,
                 config: dict,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        super(PartiallyLoadableDataset, self).__init__(input_sequence_length=input_sequence_length,
                                                       output_sequence_length=output_sequence_length,
                                                       targets_are_predictions=targets_are_predictions,
                                                       data_preprocessors=data_preprocessors,
                                                       batch_size=batch_size,
                                                       epoch_length=epoch_length,
                                                       shuffle_on_epoch_end=shuffle_on_epoch_end,
                                                       **kwargs)
        self.dataset_path = dataset_path
        self.config = copy.deepcopy(config)
        self.sub_config_index = 0
        self._shard_indices_offset = None
        self.normalization_range = None

    def sample(self, batch_size=None, seed=None, sequence_length=None, max_shard_count=1, return_labels=False):
        if batch_size is None:
            batch_size = self.batch_size

        if max_shard_count is None:
            max_shard_count = self.shards_count
        else:
            max_shard_count = min(max_shard_count, self.shards_count)

        with NumpySeedContext(seed):
            shards = self.get_random_shards(max_shard_count, return_labels)
            images, labels, labels_shard = [], [], None
            shard_size = batch_size // max_shard_count

            for i, shard in enumerate(shards):
                if return_labels:
                    images_shard, labels_shard = shard
                else:
                    images_shard = shard

                if i == (max_shard_count - 1):
                    shard_size += batch_size % max_shard_count

                indices = self.sample_indices(shard_size, len(images_shard), sequence_length)
                images.append(images_shard[indices])

                if return_labels:
                    if self.output_sequence_length is None:
                        labels_indices = indices[:, -1]
                    else:
                        labels_indices = indices[:, -self.output_sequence_length:]
                    labels.append(labels_shard[labels_indices])

        images = fast_concatenate_0(images)
        images = self.normalize_samples(images)

        if return_labels:
            # np.any(self.anomaly_labels, axis=(1, 2, 3))
            # TODO : Load pixel_level labels
            labels = fast_concatenate_0(labels)
            return images, labels, None
        else:
            return images

    def get_random_shards(self, max_shard_count, return_labels=False):
        indices = np.random.permutation(np.arange(self.shards_count))[:max_shard_count]
        shards = []
        for index in indices:
            images_shard_filepath = os.path.join(self.dataset_path, self.images_filenames[index])
            images_shard = np.load(images_shard_filepath, mmap_mode="r")
            if return_labels:
                labels_shard_filepath = os.path.join(self.dataset_path, self.labels_filenames[index])
                labels_shard = np.load(labels_shard_filepath, mmap_mode="r")
                shards.append((images_shard, labels_shard))
            else:
                shards.append(images_shard)
        return shards

    def normalize_samples(self, samples: np.ndarray):
        return samples * self.normalization_range[1] + self.normalization_range[0]

    def shuffle(self):
        pass

    def resized(self, size, input_sequence_length, output_sequence_length):
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
                             input_sequence_length=input_sequence_length,
                             output_sequence_length=output_sequence_length,
                             targets_are_predictions=self.targets_are_predictions,
                             data_preprocessors=self.data_preprocessors, batch_size=self.batch_size,
                             epoch_length=self.epoch_length, shuffle_on_epoch_end=self.shuffle_on_epoch_end)
        other.sub_config_index = target_index
        other.normalization_range = self.normalization_range
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

    @property
    def shards_count(self):
        return len(self.images_filenames)
    # endregion
