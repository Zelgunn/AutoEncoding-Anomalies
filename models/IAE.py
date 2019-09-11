import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Dict

from models import AE


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
        inputs_shape = tf.shape(inputs)
        batch_size, total_length, *dimensions = tf.unstack(inputs_shape)
        step_count = total_length // self.step_size
        inputs = tf.reshape(inputs, [batch_size * step_count, self.step_size, *dimensions])

        decoded = self.decode(self.encode(inputs))

        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def interpolate(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size, total_length, *dimensions = tf.unstack(inputs_shape)
        step_count = total_length // self.step_size
        inputs = tf.reshape(inputs, [batch_size, step_count, self.step_size, *dimensions])

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

        decoded = self.decode(encoded)
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        decoded = self.interpolate(inputs)
        reconstruction_error = tf.reduce_mean(tf.square(inputs - decoded))
        return reconstruction_error

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
