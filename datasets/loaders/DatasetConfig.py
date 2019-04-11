import numpy as np
import json
import os
from typing import Dict, Type, List, Any

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Modality, ModalityCollection, ModalityShape


def get_shard_count(sample_length: int,
                    shard_size: int
                    ) -> int:
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


class DatasetConfig(object):
    def __init__(self,
                 tfrecords_config_folder: str,
                 modalities_io_shapes: Dict[Type[Modality], ModalityShape]
                 ):
        self.tfrecords_config_folder = tfrecords_config_folder
        tf_records_config_filepath = os.path.join(tfrecords_config_folder, tfrecords_config_filename)
        with open(tf_records_config_filepath, 'r') as file:
            self.tfrecords_config: Dict[str, Any] = json.load(file)
            self.modalities = ModalityCollection.from_config(self.tfrecords_config["modalities"])
            self.subsets: Dict[str, List[str]] = self.tfrecords_config["subsets"]
            self.shard_duration: float = self.tfrecords_config["shard_duration"]

        self.modalities.set_modalities_shapes(modalities_io_shapes, filter_missing_modalities=True)

        shard_counts = []
        for modality in self.modalities.values():
            sample_length = modality.io_shape.sample_length
            shard_size = self.get_modality_max_shard_size(modality)

            shard_counts.append(get_shard_count(sample_length, shard_size))

        self.shard_count_per_sample: int = max(*shard_counts)

    def list_subset_tfrecords(self,
                              subset_name: str
                              ) -> Dict[str, List[str]]:
        subset_files = {}
        subset = self.subsets[subset_name]
        for folder in subset:
            folder = os.path.join(self.tfrecords_config_folder, folder)
            folder = os.path.normpath(folder)
            files = [file for file in os.listdir(folder) if file.endswith(".tfrecord")]
            subset_files[folder] = files
        return subset_files

    def get_modality_max_shard_size(self,
                                    modality: Modality
                                    ) -> int:
        frequency: float = modality.frequency
        shard_size = int(np.ceil(frequency * self.shard_duration))
        return shard_size
