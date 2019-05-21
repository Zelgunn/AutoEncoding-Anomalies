import tensorflow as tf
from typing import Dict, Any

from modalities import Modality


class MFCCs(Modality):
    def __init__(self, frequency: float):
        super(MFCCs, self).__init__(frequency=frequency)

    def get_config(self) -> Dict[str, Any]:
        # base_config = super(MFCCs, self).get_config()
        # config = {}
        # return {**base_config, **config}
        raise NotImplementedError

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        raise NotImplementedError

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        raise NotImplementedError

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string),
                cls.shape_id(): cls.tfrecord_shape_parse_function()}

    @classmethod
    def rank(cls) -> int:
        return 2
