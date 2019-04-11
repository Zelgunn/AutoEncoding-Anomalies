import tensorflow as tf
from typing import Dict, Optional

from modalities import Modality, ModalityShape


class RawAudio(Modality):
    def __init__(self,
                 frequency: float,
                 io_shape: Optional[ModalityShape] = None):
        super(RawAudio, self).__init__(frequency=frequency,
                                       rank=1,
                                       io_shape=io_shape)

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        raise NotImplementedError

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        raise NotImplementedError

    @classmethod
    def tfrecord_feature_parse_function(cls):
        return tf.VarLenFeature(tf.string)
