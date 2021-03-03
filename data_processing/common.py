import tensorflow as tf
from enum import Enum
from typing import Union

from misc_utils.math_utils import standardize_from


@tf.function
def normalize_sigmoid(x: tf.Tensor) -> tf.Tensor:
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


@tf.function
def normalize_tanh(x: tf.Tensor) -> tf.Tensor:
    x = normalize_sigmoid(x)
    x = tf.constant(2.0) * x - tf.constant(1.0)
    return x


class ActivationRange(Enum):
    LINEAR = 0
    SIGMOID = 1
    TANH = 2

    @staticmethod
    def from_str(s: str) -> "ActivationRange":
        s = s.lower()
        if s == "linear":
            return ActivationRange.LINEAR
        elif s == "sigmoid":
            return ActivationRange.SIGMOID
        elif s == "tanh":
            return ActivationRange.TANH
        else:
            raise ValueError("s must be in {{linear, sigmoid, tanh}} but got {}".format(s))


def apply_activation_range(x: tf.Tensor, activation_range: Union[ActivationRange, str]) -> tf.Tensor:
    if isinstance(activation_range, str):
        activation_range = ActivationRange.from_str(activation_range)

    if activation_range == activation_range.LINEAR:
        if x.shape.rank >= 3:
            x = tf.image.per_image_standardization(x)
        else:
            x = standardize_from(x, start_axis=0)
    elif activation_range == activation_range.SIGMOID:
        x = normalize_sigmoid(x)
    elif activation_range == activation_range.TANH:
        x = normalize_tanh(x)
    return x


def dropout_noise(inputs, max_rate, noise_shape_prob, seed=None):
    """
    :param inputs: A floating point `Tensor`.
    :param max_rate: A floating point scalar `Tensor`. The maximum probability that each element is dropped.
    :param noise_shape_prob: A 1-D `Tensor` of type `float32`, representing the probability of dropping each
        dimension completely.
    :param seed: A Python integer. Used to create a random seed for the distribution.
    :return: A `Tensor` of the same shape and type as `inputs.
    """
    noise_shape = tf.random.uniform(shape=[len(inputs.shape)], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    noise_shape = noise_shape < noise_shape_prob
    noise_shape = tf.cast(noise_shape, tf.int32)
    noise_shape = tf.shape(inputs) * (1 - noise_shape) + noise_shape

    rate = tf.random.uniform(shape=[], minval=0.0, maxval=max_rate, dtype=tf.float32, seed=seed)
    random_tensor = tf.random.uniform(shape=noise_shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    keep_mask = random_tensor >= rate

    outputs = inputs * tf.cast(keep_mask, inputs.dtype) / (1.0 - rate)
    return outputs
