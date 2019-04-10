import numpy as np
import json
from typing import Dict, Union, List, Tuple


def get_shard_count(sample_length, shard_size):
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


class DatasetConfig(object):
    def __init__(self,
                 dataset_records_config_filepath: str,
                 model_inputs_shapes_by_modality: Dict[str, Union[List[int], Tuple[int]]]):
        with open(dataset_records_config_filepath, 'r') as file:
            self.dataset_records_config = json.load(file)
            self.dataset_modalities_config = self.dataset_records_config["modalities"]

        self.model_inputs_shapes_by_modality = model_inputs_shapes_by_modality
        self.used_modalities_names = list(self.model_inputs_shapes_by_modality.keys())

        shard_counts = []
        for modality in self.used_modalities_names:
            sample_length = self.model_inputs_shapes_by_modality[modality][0]
            shard_size = self.dataset_modalities_config[modality]["shard_size"]
            shard_counts.append(get_shard_count(sample_length, shard_size))

        self.shard_count_per_sample = max(*shard_counts)
