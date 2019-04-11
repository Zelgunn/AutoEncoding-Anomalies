import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Any, Union

from modalities import Modality, ModalityShape


class OpticalFlow(Modality):
    def __init__(self,
                 frequency: float,
                 use_polar: bool,
                 pyr_scale=0.5,
                 levels=3,
                 winsize=5,
                 iterations=5,
                 poly_n=5,
                 poly_sigma=1.2,
                 io_shape: Optional[ModalityShape] = None):
        super(OpticalFlow, self).__init__(frequency=frequency,
                                          rank=4,
                                          io_shape=io_shape)
        self.use_polar = use_polar
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def get_config(self) -> Dict[str, Any]:
        base_config = super(OpticalFlow, self).get_config()
        config = \
            {
                "use_polar": self.use_polar,
                **self.farneback_params
            }
        return {**base_config, **config}

    @property
    def farneback_params(self) -> Dict[str, Union[float, int]]:
        return {
            "pyr_scale": self.pyr_scale,
            "levels": self.levels,
            "winsize": self.winsize,
            "iterations": self.iterations,
            "poly_n": self.poly_n,
            "poly_sigma": self.poly_sigma
        }

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        return cls.encode_raw(modality_value, np.float16)

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        return cls.decode_raw(parsed_features, tf.float16)

    @classmethod
    def tfrecord_feature_parse_function(cls):
        return tf.VarLenFeature(tf.string)
