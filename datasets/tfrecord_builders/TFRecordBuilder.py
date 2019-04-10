import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json
from typing import Union, Tuple, List, Dict, Any

from datasets.modality_builders import AdaptiveBuilder, VideoReader, AudioReader
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


dataset_tfrecords_info_filename = "dataset_tfrecords_info.json"


class TFRecordBuilder(object):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 modalities: Dict[str, Dict[str, Any]],
                 verbose=1):

        self.dataset_path = dataset_path
        self.verbose = verbose
        self.shard_duration = shard_duration
        self.modalities = modalities

        self.parse_modalities_frequency()

    # region Build
    def get_dataset_sources(self) -> List[DataSource]:
        raise NotImplementedError("`get_dataset_sources` should be defined for subclasses.")

    def build(self):
        data_sources = self.get_dataset_sources()

        dataset_info_dict: Dict[str, Union[List[str], Dict]] = {"modalities": self.modalities}

        for data_source in data_sources:
            if self.verbose > 0:
                print("Building {}".format(data_source.target_path))

            target_path = os.path.relpath(data_source.target_path, self.dataset_path)
            if data_source.subset_name in dataset_info_dict:
                dataset_info_dict[data_source.subset_name].append(target_path)
            else:
                dataset_info_dict[data_source.subset_name] = [target_path]

            self.build_one(data_source)

        with open(os.path.join(self.dataset_path, dataset_tfrecords_info_filename), 'w') as file:
            json.dump(dataset_info_dict, file)

    def build_one(self, data_source: Union[DataSource, List[DataSource]]):
        modality_builder = AdaptiveBuilder(shard_duration=self.shard_duration,
                                           modalities=self.modalities,
                                           video_source=data_source.video_source,
                                           video_frame_size=data_source.video_frame_size,
                                           audio_source=data_source.audio_source)

        # TODO : Give frequency source instead of static "raw_video"
        shard_count = modality_builder.get_shard_count()
        labels_iterator = LabelsBuilder(data_source.labels_source,
                                        shard_count=shard_count,
                                        shard_duration=self.shard_duration,
                                        frequency=modality_builder.get_modality_frequency("raw_video"))

        source_iterator = zip(modality_builder, labels_iterator)

        for i, shard in enumerate(source_iterator):
            if self.verbose > 0:
                print("\r{} : {}/{}".format(data_source.target_path, i, shard_count), end='')
            sys.stdout.flush()

            modalities, labels = shard

            features = {}
            for modality_name, modality_value in modalities.items():
                if modality_name == "raw_video":
                    if modality_value.ndim == 3:
                        modality_value = np.expand_dims(modality_value, axis=-1)
                    features[modality_name] = video_feature(modality_value)

                elif modality_name in ["flow", "dog"]:
                    set_ndarray_feature(features, modality_name, modality_value.astype("float16"))

                elif modality_name in ["mfcc"]:
                    set_ndarray_feature(features, modality_name, modality_value)

                elif modality_name == "raw_audio":
                    features[modality_name] = float_list_feature(modality_value)

                else:
                    raise NotImplementedError("{} is not implemented.".format(modality_name))

            features["labels"] = float_list_feature(labels)

            example = tf.train.Example(features=tf.train.Features(feature=features))

            shard_filepath = os.path.join(data_source.target_path, "shard_{:05d}.tfrecord".format(i))
            writer = tf.io.TFRecordWriter(shard_filepath)
            writer.write(example.SerializeToString())

        if self.verbose > 0:
            print("\r{} : Done".format(data_source.target_path))

    # endregion

    def parse_modalities_frequency(self):
        for modality in self.modalities:
            assert "frequency" in self.modalities[modality], \
                "modalities dictionary must give `frequency` for each modality"

            frequency = self.modalities[modality]["frequency"]
            if isinstance(frequency, str):
                names_met = []
                while isinstance(frequency, str):
                    names_met.append(frequency)
                    frequency = self.modalities[frequency]["frequency"]

                    if frequency in names_met:
                        raise ValueError("Cyclic reference in modalities frequencies : " +
                                         " -> ".join(names_met) + " -> " + frequency)

                self.modalities[modality]["frequency"] = frequency


def video_feature(video):
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_frames))


def set_ndarray_feature(features: Dict, name: str, array: np.ndarray):
    features[name + "_shape"] = int64_list_feature(array.shape)
    features[name] = bytes_list_feature([array.tobytes()])


def int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def float_list_feature(float_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))


def bytes_list_feature(bytes_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))
