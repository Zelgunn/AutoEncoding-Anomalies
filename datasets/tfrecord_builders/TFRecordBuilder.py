import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from typing import Union, Tuple, List, Dict, Any

from datasets.modality_builders import AdaptiveBuilder, LabelsBuilder, VideoReader, AudioReader


class DataSource(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[int, int]]],
                 target_path: str,
                 video_source: Union[VideoReader, str, cv2.VideoCapture, np.ndarray, List[str]] = None,
                 video_frame_size: Tuple[int, int] = None,
                 audio_source: Union[AudioReader, str, np.ndarray] = None
                 ):
        self.labels_source = labels_source
        self.target_path = target_path

        self.video_source = video_source
        self.video_frame_size = video_frame_size
        self.audio_source = audio_source


class TFRecordBuilder(object):
    def __init__(self,
                 dataset_path: str,
                 modalities: Dict[str, Dict[str, Any]],
                 verbose=1):
        self.dataset_path = dataset_path
        self.modalities = modalities
        self.verbose = verbose

    def build(self, data_source: Union[DataSource, List[DataSource]]):
        if isinstance(data_source, list):
            for source in data_source:
                if self.verbose > 0:
                    print("Building {}".format(source.target_path))
                self.build(source)
            return

        modality_builder = AdaptiveBuilder(self.modalities,
                                           data_source.video_source,
                                           data_source.video_frame_size,
                                           data_source.audio_source)
        # TODO : Change the way shard_size is selected
        labels_iterator = LabelsBuilder(data_source.labels_source,
                                        shard_size=modality_builder.get_shard_size("raw_video"),
                                        shard_count=modality_builder.get_shard_count())

        source_iterator = zip(modality_builder, labels_iterator)

        for i, shard in enumerate(source_iterator):
            if self.verbose > 0:
                print("\r{} : {}/{}".format(data_source.target_path,
                                            i,
                                            labels_iterator.shard_count), end='')
            sys.stdout.flush()

            modalities, labels = shard

            features = {}
            for modality_name, modality_value in modalities.items():
                if modality_name in ["raw_video", "dog"]:
                    if modality_value.ndim == 3:
                        modality_value = np.expand_dims(modality_value, axis=-1)
                    features[modality_name] = video_feature(modality_value)

                elif modality_name == "flow":
                    flow_x, flow_y = np.split(modality_value, indices_or_sections=2, axis=-1)
                    features["flow_x"] = video_feature(flow_x)
                    features["flow_y"] = video_feature(flow_y)

                elif modality_name == "raw_audio":
                    features["raw_audio"] = float_list_feature(modality_value)

                elif modality_name == "mfcc":
                    features["mfcc"] = mfcc_feature(modality_value)

            features["labels"] = float_list_feature(labels)

            example = tf.train.Example(features=tf.train.Features(feature=features))

            shard_filepath = os.path.join(data_source.target_path, "shard_{:5d}.tfrecord".format(i))
            writer = tf.io.TFRecordWriter(shard_filepath)
            writer.write(example.SerializeToString())

        if self.verbose > 0:
            print("\r{} : Done".format(data_source.target_path))


def video_feature(video):
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_frames))


def float_list_feature(float_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))


def mfcc_feature(mfcc):
    mfcc_as_bytes_list = [mfcc[i].tobytes for i in range(len(mfcc))]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=mfcc_as_bytes_list))
