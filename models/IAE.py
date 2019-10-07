import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from models import AE
from models.utils import split_steps


class IAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 step_size: int,
                 learning_rate=1e-3,
                 **kwargs):
        super(IAE, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  learning_rate=learning_rate,
                                  **kwargs)
        self.step_size = step_size

    def call(self, inputs, training=None, mask=None):
        inputs, inputs_shape, new_shape = self.split_inputs(inputs, merge_batch_and_steps=True)
        decoded = self.decode(self.encode(inputs))
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        return self.reconstruction_error(inputs)

    def compute_combined_errors(self, inputs, metric):
        encoding_delta = self.encoding_error(inputs)
        reconstruction_error = self.reconstruction_error(inputs,
                                                         metric=metric,
                                                         reduction_axis=tuple(range(2, inputs.shape.rank)))
        return encoding_delta * 0.1 + reconstruction_error

    def compute_combined_errors_mse(self, inputs):
        return self.compute_combined_errors(inputs, metric="mse")

    def compute_combined_errors_mae(self, inputs):
        return self.compute_combined_errors(inputs, metric="mae")

    def reconstruction_error(self, inputs, metric="mse", reduction_axis=None):
        decoded = self.interpolate(inputs)

        if metric == "mse":
            error = tf.square(inputs - decoded)
        elif metric == "mae":
            error = tf.abs(inputs - decoded)
        else:
            raise ValueError("Unknown metric : {}".format(metric))

        reconstruction_error = tf.reduce_mean(error, axis=reduction_axis)
        return reconstruction_error

    @tf.function
    def encoding_error(self, inputs):
        target = self.get_interpolated_latent_code(inputs)

        inputs, _, new_shape = self.split_inputs(inputs, merge_batch_and_steps=True)
        pred = self.encode(inputs)

        batch_size, step_count, *_ = new_shape
        new_shape = [batch_size, step_count * self.step_size, -1]
        target = tf.reshape(target, new_shape)
        pred = tf.reshape(pred, new_shape)

        # error = tf.reduce_mean(tf.square(target - pred), axis=-1)
        error = 1 + tf.losses.cosine_similarity(target, pred, axis=-1)
        return error

    @tf.function
    def interpolate(self, inputs):
        inputs_shape = tf.shape(inputs)
        encoded = self.get_interpolated_latent_code(inputs)
        decoded = self.decode(encoded)
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def get_interpolated_latent_code(self, inputs):
        inputs, _, new_shape = self.split_inputs(inputs, merge_batch_and_steps=False)
        batch_size, step_count, *_ = new_shape

        encoded_first = self.encode(inputs[:, 0])
        encoded_last = self.encode(inputs[:, -1])

        encoded_shape_dimensions = tf.unstack(tf.shape(encoded_first)[1:])
        tile_multiples = [1, step_count] + [1] * (len(inputs.shape) - 2)
        encoded_first = tf.tile(tf.expand_dims(encoded_first, axis=1), tile_multiples)
        encoded_last = tf.tile(tf.expand_dims(encoded_last, axis=1), tile_multiples)

        weights = tf.linspace(0.0, 1.0, step_count)
        weights = tf.reshape(weights, tile_multiples)

        encoded = encoded_first * (1.0 - weights) + encoded_last * weights
        encoded = tf.reshape(encoded, [batch_size * step_count, *encoded_shape_dimensions])
        return encoded

    @tf.function
    def split_inputs(self, inputs, merge_batch_and_steps):
        return split_steps(inputs, self.step_size, merge_batch_and_steps)

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "step_count": self.step_size,
            "learning_rate": self.learning_rate,
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.encoder: "encoder",
                self.decoder: "decoder"}
