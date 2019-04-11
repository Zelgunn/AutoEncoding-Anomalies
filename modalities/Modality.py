import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Optional, Any

from modalities import int64_list_feature, bytes_list_feature


class ModalityShape(object):
    def __init__(self,
                 input_shape: Union[List[int], Tuple[int, ...]],
                 output_shape: Union[List[int], Tuple[int, ...]]):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def input_length(self) -> int:
        return self.input_shape[0]

    @property
    def output_length(self) -> int:
        return self.output_shape[0]

    @property
    def sample_length(self) -> int:
        return max(self.input_length, self.output_length)

    @property
    def rank(self) -> int:
        return len(self.input_shape)


class Modality(ABC):
    def __init__(self, frequency: float):
        self.frequency = frequency
        self.io_shape: Optional[ModalityShape] = None

    def get_config(self) -> Dict[str, Any]:
        config = {"frequency": self.frequency}
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    @classmethod
    @abstractmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        raise NotImplementedError

    @classmethod
    def tfrecord_shape_parse_function(cls):
        return tf.FixedLenFeature([cls.rank()], dtype=tf.int64)

    @classmethod
    def encode_raw(cls, array: np.ndarray, dtype: Union[type, str]) -> Dict[str, tf.train.Feature]:
        features = {
            cls.tfrecord_shape_id(): int64_list_feature(array.shape),
            cls.tfrecord_id(): bytes_list_feature([array.astype(dtype).tobytes()])
        }
        return features

    @classmethod
    def decode_raw(cls, parsed_features: Dict[str, tf.SparseTensor], dtype: tf.dtypes.DType):
        encoded_modality = parsed_features[cls.tfrecord_id()].values
        modality_shape = parsed_features[cls.tfrecord_shape_id()]

        decoded_modality = tf.decode_raw(encoded_modality, dtype)
        decoded_modality = tf.reshape(decoded_modality, modality_shape)
        return decoded_modality

    @classmethod
    def tfrecord_id(cls):
        return cls.__name__

    @classmethod
    def tfrecord_shape_id(cls):
        return cls.tfrecord_id() + "_shape"

    @classmethod
    @abstractmethod
    def rank(cls) -> int:
        raise NotImplementedError
