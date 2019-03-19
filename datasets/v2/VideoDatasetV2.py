import os
from typing import Dict
import json

from datasets import DatasetV2, VideoSubsetV2, DatasetConfigV2


class VideoDatasetV2(DatasetV2):
    def __init__(self, dataset_path: str, config: DatasetConfigV2):
        super(VideoDatasetV2, self).__init__(dataset_path=dataset_path,
                                             config=config)

        with open(os.path.join(dataset_path, "header.json"), 'r') as file:
            header = json.load(file)
            self.config.shard_size = header["shard_size"]

        potential_subsets = []
        for element in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, element)):
                potential_subsets.append(element)

        subsets_dict = {}
        for potential_subset in potential_subsets:
            subset_path = os.path.join(dataset_path, potential_subset)
            for video_dir in os.listdir(subset_path):
                video_path = os.path.join(subset_path, video_dir)
                if os.path.isdir(video_path):
                    video_shards = os.listdir(video_path)
                    video_shards = [shard for shard in video_shards if shard.endswith(".tfrecord")]
                    if len(video_shards) > 0:
                        if potential_subset in subsets_dict:
                            subsets_dict[potential_subset][video_path] = video_shards
                        else:
                            subsets_dict[potential_subset] = {video_path: video_shards}

        self.subsets: Dict[str: VideoSubsetV2] = {}
        for subset_name in subsets_dict:
            self.subsets[subset_name.lower()] = VideoSubsetV2(subsets_dict[subset_name], self.config)

    @property
    def train_subset(self) -> VideoSubsetV2:
        subset: VideoSubsetV2 = super(VideoDatasetV2, self).train_subset
        return subset

    @property
    def test_subset(self) -> VideoSubsetV2:
        subset: VideoSubsetV2 = super(VideoDatasetV2, self).test_subset
        return subset
