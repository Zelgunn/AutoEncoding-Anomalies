import numpy as np
import json
import os
from typing import Dict, Type, List, Any, Tuple, Union

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Modality, ModalityCollection, ModalityShape
from modalities import RawVideo, OpticalFlow, DoG, Landmarks
from modalities import RawAudio, MelSpectrogram
from utils.misc_utils import int_ceil


def get_shard_count(sample_length: int,
                    shard_size: int
                    ) -> int:
    shard_count = 1 + int_ceil((sample_length - 1) / shard_size)
    return max(2, shard_count)


class DatasetConfig(object):
    def __init__(self,
                 tfrecords_config_folder: str,
                 modalities_io_shapes: Dict[Type[Modality], ModalityShape],
                 output_range: Tuple[float, float],
                 ):
        self.tfrecords_config_folder = tfrecords_config_folder
        tf_records_config_filepath = os.path.join(tfrecords_config_folder, tfrecords_config_filename)
        with open(tf_records_config_filepath, 'r') as file:
            self.tfrecords_config: Dict[str, Any] = json.load(file)

        self.modalities = ModalityCollection.from_config(self.tfrecords_config["modalities"])
        self.subsets: Dict[str, List[str]] = self.tfrecords_config["subsets"]
        self.shard_duration = float(self.tfrecords_config["shard_duration"])
        self.video_frequency = self.tfrecords_config["video_frequency"]
        self.audio_frequency = self.tfrecords_config["audio_frequency"]
        self.max_labels_size: int = int(self.tfrecords_config["max_labels_size"])

        self.modalities_ranges = self.tfrecords_config["modalities_ranges"]

        self.modalities.set_modalities_shapes(modalities_io_shapes, filter_missing_modalities=True)
        self.output_range = output_range

        # region Compute maximum amount of shards required to build a sample
        shard_counts = []
        for modality in self.modalities:
            sample_length = modality.io_shape.sample_length
            shard_size = self.get_modality_max_shard_size(modality)

            shard_counts.append(get_shard_count(sample_length, shard_size))

        self.shards_per_sample: int = max(shard_counts)
        # endregion

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

    def get_modality_shard_size(self,
                                modality: Modality
                                ) -> Union[float, int]:

        if isinstance(modality, (RawVideo, DoG, Landmarks)):
            shard_size = self.video_frequency * self.shard_duration
        elif isinstance(modality, OpticalFlow):
            shard_size = self.video_frequency * self.shard_duration - 1
        elif isinstance(modality, RawAudio):
            shard_size = self.audio_frequency * self.shard_duration
        elif isinstance(modality, MelSpectrogram):
            shard_size = modality.get_output_frame_count(self.shard_duration * self.audio_frequency,
                                                         self.audio_frequency)
        else:
            raise NotImplementedError(modality.id())

        return shard_size

    def get_modality_max_shard_size(self, modality: Modality) -> int:
        return int_ceil(self.get_modality_shard_size(modality))
