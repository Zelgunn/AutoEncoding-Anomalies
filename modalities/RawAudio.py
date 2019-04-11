import tensorflow as tf
from typing import Dict

from modalities import Modality


class RawAudio(Modality):
    def __init__(self, frequency: float):
        super(RawAudio, self).__init__(frequency=frequency)

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        raise NotImplementedError

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        raise NotImplementedError

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.tfrecord_id(): tf.VarLenFeature(tf.string)}

    @classmethod
    def rank(cls) -> int:
        return 1
