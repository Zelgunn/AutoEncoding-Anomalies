import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Any

from modalities import Modality, ModalityShape


class DoG(Modality):
    def __init__(self,
                 frequency: float,
                 blurs=(2.0, 2.82, 4.0, 5.66, 8.0),
                 io_shape: Optional[ModalityShape] = None):
        super(DoG, self).__init__(frequency=frequency,
                                  rank=4,
                                  io_shape=io_shape)
        self.blurs = blurs

    def get_config(self) -> Dict[str, Any]:
        base_config = super(DoG, self).get_config()
        config = {"blurs": list(self.blurs)}
        return {**base_config, **config}

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        return cls.encode_raw(modality_value, np.float16)

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        return cls.decode_raw(parsed_features, tf.float16)

    @classmethod
    def tfrecord_feature_parse_function(cls):
        return tf.VarLenFeature(tf.string)
