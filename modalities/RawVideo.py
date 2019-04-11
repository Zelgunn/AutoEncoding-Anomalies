import tensorflow as tf
import cv2
import numpy as np
from typing import Dict

from modalities import Modality


class RawVideo(Modality):
    def __init__(self, frequency: float):
        super(RawVideo, self).__init__(frequency=frequency,
                                       rank=4)

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        if modality_value.ndim == 3:
            modality_value = np.expand_dims(modality_value, axis=-1)

        return {cls.tfrecord_id(): video_feature(modality_value)}

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        # TODO : Type parsed_features
        print(type(parsed_features))
        exit()

        # TODO : Type encoded_raw_video
        encoded_raw_video = parsed_features[cls.tfrecord_id()].values
        print(type(encoded_raw_video))
        exit()

        raw_video_shard_size = tf.shape(encoded_raw_video)[0]
        raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_raw_video[i]), tf.float32),
                              tf.range(raw_video_shard_size),
                              dtype=tf.float32)
        # raw_video_pad_size = max_shard_size - raw_video_shard_size
        #
        # def images_padded():
        #     paddings = [[0, raw_video_pad_size], [0, 0], [0, 0], [0, 0]]
        #     return tf.pad(raw_video, paddings)
        #
        # def images_identity():
        #     return raw_video
        #
        # raw_video = tf.cond(pred=raw_video_pad_size > 0,
        #                     true_fn=images_padded,
        #                     false_fn=images_identity)

        return raw_video

    @classmethod
    def tfrecord_feature_parse_function(cls):
        return tf.VarLenFeature(tf.string)


def video_feature(video):
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_frames))
