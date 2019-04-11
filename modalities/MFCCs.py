import tensorflow as tf
from typing import Dict, Optional, Any

from modalities import Modality, ModalityShape


class MFCCs(Modality):
    def __init__(self,
                 frequency: float,
                 io_shape: Optional[ModalityShape] = None):
        super(MFCCs, self).__init__(frequency=frequency,
                                    rank=2,
                                    io_shape=io_shape)

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
    def tfrecord_feature_parse_function(cls):
        return tf.VarLenFeature(tf.string)
