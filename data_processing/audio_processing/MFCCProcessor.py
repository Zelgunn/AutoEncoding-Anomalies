import tensorflow as tf
from typing import Union

from data_processing import DataProcessor
from data_processing.common import ActivationRange
from misc_utils.math_utils import standardize_from


class MFCCProcessor(DataProcessor):
    def __init__(self, activation_range: Union[str, ActivationRange]):
        if isinstance(activation_range, str):
            activation_range = ActivationRange.from_str(activation_range)
        self.activation_range: ActivationRange = activation_range

        self.min_value = tf.Variable(initial_value=1000.0, trainable=False, name="MFCC_min_value")
        self.max_value = tf.Variable(initial_value=-1000.0, trainable=False, name="MFCC_max_value")
        self.allow_update = tf.Variable(initial_value=True, trainable=False, name="MFCC_allow_update")

    def reset(self):
        self.min_value.assign(1000.0)
        self.max_value.assign(-1000.0)
        self.allow_update.assign(True)

    @tf.function
    def enable_updates(self):
        self.allow_update.assign(True)

    @tf.function
    def disable_updates(self):
        self.allow_update.assign(False)

    @tf.function
    def update(self, min_value, max_value):
        def _update():
            self.min_value.assign(min_value)
            self.max_value.assign(max_value)
            return self.min_value, self.max_value

        def _do_nothing():
            return min_value, max_value

        min_value, max_value = tf.cond(self.allow_update, _update, _do_nothing)
        return min_value, max_value

    @tf.function
    def pre_process(self, inputs: tf.Tensor) -> tf.Tensor:
        min_value = tf.minimum(self.min_value, tf.reduce_min(inputs))
        max_value = tf.maximum(self.max_value, tf.reduce_max(inputs))

        min_value, max_value = self.update(min_value, max_value)

        if self.activation_range in [ActivationRange.SIGMOID, ActivationRange.TANH]:
            inputs = (inputs - min_value) / (max_value - min_value)

        if self.activation_range == ActivationRange.TANH:
            inputs = inputs * 2.0 - 1.0

        if self.activation_range == ActivationRange.LINEAR:
            inputs = standardize_from(inputs, start_axis=0)

        return inputs

    @tf.function
    def post_process(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.activation_range == ActivationRange.LINEAR:
            current_min = tf.reduce_min(inputs)
            current_max = tf.reduce_max(inputs)
            inputs = (inputs - current_min) / (current_max - current_min)

        if self.activation_range == ActivationRange.TANH:
            # noinspection PyTypeChecker
            inputs: tf.Tensor = (inputs + 1.0) * 0.5

        inputs = inputs * (self.max_value - self.min_value) + self.min_value
        return inputs
