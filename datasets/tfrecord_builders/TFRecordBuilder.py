import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json
from typing import Union, Tuple, List, Dict, Type, Optional

from modalities import Modality, ModalityCollection
from modalities.modality_utils import float_list_feature
from datasets.modality_builders import ModalityBuilder, VideoBuilder, AudioBuilder, BuildersList
from datasets.data_readers import VideoReader, AudioReader
from datasets.labels_builders import LabelsBuilder


class DataSource(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[float, float]]],
                 target_path: str,
                 subset_name: str,
                 video_source: Union[VideoReader, str, cv2.VideoCapture, np.ndarray, List[str]] = None,
                 video_frame_size: Tuple[int, int] = None,
                 audio_source: Union[AudioReader, str, np.ndarray] = None
                 ):
        self.labels_source = labels_source
        self.target_path = target_path
        self.subset_name = subset_name

        self.video_source = video_source
        self.video_frame_size = video_frame_size
        self.audio_source = audio_source


tfrecords_config_filename = "tfrecords_config.json"


class TFRecordBuilder(object):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 labels_frequency: Union[int, float] = None,
                 verbose=1):

        self.dataset_path = dataset_path
        self.shard_duration = shard_duration
        if video_frequency is None and audio_frequency is None:
            raise ValueError("You must specify at least the frequency for either Video or Audio, got None and None")
        self.video_frequency = video_frequency
        self.audio_frequency = audio_frequency
        self.labels_frequency = labels_frequency
        self.modalities = modalities
        self.verbose = verbose

    def get_dataset_sources(self) -> List[DataSource]:
        raise NotImplementedError("`get_dataset_sources` should be defined for subclasses.")

    def build(self):
        data_sources = self.get_dataset_sources()

        subsets_dict: Dict[str, Union[List[str], Dict]] = {}

        min_values = None
        max_values = None
        max_labels_sizes = []

        for i, data_source in enumerate(data_sources):
            if self.verbose > 0:
                print("Building {}/{} - {}".format(i + 1, len(data_sources), data_source.target_path))

            # region Fill subsets_dict with folders containing shards
            target_path = os.path.relpath(data_source.target_path, self.dataset_path)
            if data_source.subset_name in subsets_dict:
                subsets_dict[data_source.subset_name].append(target_path)
            else:
                subsets_dict[data_source.subset_name] = [target_path]
            # endregion

            source_min_values, source_max_values, max_labels_size = self.build_one(data_source)

            # region Modalities min/max for normalization (step 1 : get)
            if min_values is None:
                min_values = source_min_values
                max_values = source_max_values
            else:
                for modality_type in source_min_values:
                    min_values[modality_type] += source_min_values[modality_type]
                    max_values[modality_type] += source_max_values[modality_type]
            # endregion

            max_labels_sizes.append(max_labels_size)

        # region Modalities min/max for normalization (step 2 : compute)
        modalities_ranges = {
            modality_type.id(): [float(min(min_values[modality_type])),
                                 float(max(max_values[modality_type]))]
            for modality_type in min_values
        }
        # endregion
        max_labels_size = max(max_labels_sizes)

        tfrecords_config = {
            "modalities": self.modalities.get_config(),
            "shard_duration": self.shard_duration,
            "video_frequency": self.video_frequency,
            "audio_frequency": self.audio_frequency,
            "subsets": subsets_dict,
            "modalities_ranges": modalities_ranges,
            "max_labels_size": max_labels_size,
        }

        with open(os.path.join(self.dataset_path, tfrecords_config_filename), 'w') as file:
            json.dump(tfrecords_config, file)

    def build_one(self, data_source: Union[DataSource, List[DataSource]]):
        builders = self.make_builders(video_source=data_source.video_source,
                                      video_frame_size=data_source.video_frame_size,
                                      audio_source=data_source.audio_source)

        modality_builder = BuildersList(builders=builders)

        # TODO : Delete previous .tfrecords
        shard_count = modality_builder.get_shard_count()
        labels_iterator = LabelsBuilder(data_source.labels_source,
                                        shard_count=shard_count,
                                        shard_duration=self.shard_duration,
                                        frequency=self.labels_frequency)

        source_iterator = zip(modality_builder, labels_iterator)

        min_values = {}
        max_values = {}
        max_labels_size = 0

        for i, shard in enumerate(source_iterator):
            # region Verbose
            if self.verbose > 0:
                print("\r{} : {}/{}".format(data_source.target_path, i, shard_count), end='')
            sys.stdout.flush()
            # endregion

            modalities, labels = shard
            modalities: Dict[Type[Modality], np.ndarray] = modalities

            features: Dict[str, tf.train.Feature] = {}
            for modality_type, modality_value in modalities.items():
                # region Encode modality
                encoded_features = modality_type.encode_to_tfrecord_feature(modality_value)
                for feature_name, encoded_feature in encoded_features.items():
                    features[feature_name] = encoded_feature
                # endregion

                # region Get modality min/max for normalization
                modality_min = modality_value.min()
                modality_max = modality_value.max()
                if modality_type not in min_values:
                    min_values[modality_type] = [modality_min]
                    max_values[modality_type] = [modality_max]
                else:
                    min_values[modality_type] += [modality_min]
                    max_values[modality_type] += [modality_max]
                # endregion

            max_labels_size = max(max_labels_size, len(labels))
            features["labels"] = float_list_feature(labels)

            example = tf.train.Example(features=tf.train.Features(feature=features))

            # region Write to disk
            shard_filepath = os.path.join(data_source.target_path, "shard_{:05d}.tfrecord".format(i))
            writer = tf.io.TFRecordWriter(shard_filepath)
            writer.write(example.SerializeToString())
            # endregion

        if self.verbose > 0:
            print("\r{} : Done".format(data_source.target_path))

        return min_values, max_values, max_labels_size

    def make_builders(self,
                      video_source: Union[VideoReader, str, cv2.VideoCapture, np.ndarray, List[str]],
                      video_frame_size: Tuple[int, int],
                      audio_source: Union[AudioReader, str, np.ndarray],
                      ):

        builders: List[ModalityBuilder] = []

        if VideoBuilder.supports_any(self.modalities):
            video_builder = VideoBuilder(shard_duration=self.shard_duration,
                                         source_frequency=self.video_frequency,
                                         modalities=self.modalities,
                                         video_reader=video_source,
                                         default_frame_size=video_frame_size)
            builders.append(video_builder)

        if AudioBuilder.supports_any(self.modalities):
            audio_builder = AudioBuilder(shard_duration=self.shard_duration,
                                         source_frequency=self.audio_frequency,
                                         modalities=self.modalities,
                                         audio_reader=audio_source)
            builders.append(audio_builder)

        return builders
