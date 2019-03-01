import numpy as np
import copy
import os
from typing import List

from datasets import Dataset
from data_preprocessors import DataPreprocessor


class PartiallyLoadableDataset(Dataset):
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 dataset_path: str,
                 config: dict,
                 epoch_length: int,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        super(PartiallyLoadableDataset, self).__init__(input_sequence_length=input_sequence_length,
                                                       output_sequence_length=output_sequence_length,
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

    def sample_single_video(self, video_index, shard_size, sequence_length, return_labels):
        video_shard = self.get_video_hook(video_index)

        indices = self.sample_indices(shard_size, len(video_shard), sequence_length)
        video = video_shard[indices]
        video = self.normalize_samples(video)

        video_labels = None
        if return_labels:
            shard_labels = self.get_labels_hook(video_index)
            labels_indices = indices[:, -self.output_sequence_length:]
            video_labels = shard_labels[labels_indices]

        # np.any(self.anomaly_labels, axis=(1, 2, 3))
        # TODO : Load pixel_level labels

        return video, video_labels, None

    def normalize_samples(self, samples: np.ndarray):
        return samples * self.normalization_range[1] + self.normalization_range[0]

    def get_video_hook(self, video_index):
        video_shard_filepath = os.path.join(self.dataset_path, self.videos_filenames[video_index])
        return np.load(video_shard_filepath, mmap_mode="r")

    def get_labels_hook(self, video_index):
        shard_labels_filepath = os.path.join(self.dataset_path, self.labels_filenames[video_index])
        return np.load(shard_labels_filepath, mmap_mode="r")

    def get_video_length(self, video_index):
        return len(self.get_video_hook(video_index))

    def get_video_frames(self, video_index, start, end):
        video_shard = self.get_video_hook(video_index)
        return video_shard[start:end]

    def get_video_frame_labels(self, video_index, start, end):
        labels_shard = self.get_labels_hook(video_index)
        return labels_shard[start:end]

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
    def videos_count(self):
        return len(self.videos_filenames)

    def samples_count(self) -> int:
        return self.sub_config["samples_count"]

    @property
    def images_size(self):
        return self.sub_config["images_size"]

    @property
    def videos_filenames(self):
        return self.sub_config["videos_filenames"]

    @property
    def labels_filenames(self):
        return self.sub_config["labels_filenames"]

    @property
    def shards_count(self):
        return len(self.videos_filenames)
    # endregion
