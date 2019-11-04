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
        # decoded = self.interpolate(inputs)
        start = inputs[:, :self.step_size]
        end = inputs[:, -self.step_size:]

        step_count = tf.shape(inputs)[1]
        max_offset = step_count - self.step_size
        offset = tf.random.uniform(shape=[], minval=0, maxval=max_offset + 1, dtype=tf.int32)
        target = inputs[:, offset:offset + self.step_size]

        factor = tf.cast(offset / max_offset, tf.float32)
        start_encoded = self.encode(start)
        end_encoded = self.encode(end)
        latent_code = factor * end_encoded + (1.0 - factor) * start_encoded
        decoded = self.decode(latent_code)

        loss = tf.square(target - decoded)
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)
            loss_scaled = loss * tf.cast(tf.reduce_prod(tf.shape(inputs)[1:]), tf.float32)

        gradients = tape.gradient(loss_scaled, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # loss /= tf.cast(tf.reduce_prod(tf.shape(inputs)[1:]), tf.float32)
        return loss

    @tf.function
    def interpolate(self, inputs):
        inputs_shape = tf.shape(inputs)
        encoded = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        decoded = self.decode(encoded)
        decoded = tf.reshape(decoded, inputs_shape)
        return decoded

    @tf.function
    def interpolation_mse(self, inputs):
        return self.interpolation_error(inputs, tf.losses.mse)

    @tf.function
    def interpolation_mae(self, inputs):
        return self.interpolation_error(inputs, tf.losses.mae)

    def interpolation_error(self, inputs, metric):
        interpolated = self.interpolate(inputs)

        inputs = inputs[:, self.step_size: - self.step_size]
        interpolated = interpolated[:, self.step_size: - self.step_size]

        error = metric(inputs, interpolated)
        error = tf.reduce_mean(error, axis=list(range(2, error.shape.rank)))
        return error

    @tf.function
    def interpolation_relative_mse(self, inputs):
        return self.interpolation_relative_error(inputs, tf.losses.mse)

    @tf.function
    def interpolation_relative_mae(self, inputs):
        return self.interpolation_relative_error(inputs, tf.losses.mae)

    def interpolation_relative_error(self, inputs, metric):
        base_error = metric(inputs, self(inputs))
        base_error = tf.reduce_mean(base_error, axis=list(range(2, base_error.shape.rank)))

        interpolation_error = metric(inputs, self.interpolate(inputs))
        interpolation_error = tf.reduce_mean(interpolation_error, axis=list(range(2, interpolation_error.shape.rank)))

        relative_error = tf.abs(base_error - interpolation_error)
        return relative_error

    def get_interpolated_latent_code(self, inputs, merge_batch_and_steps):
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
        if merge_batch_and_steps:
            encoded = tf.reshape(encoded, [batch_size * step_count, *encoded_shape_dimensions])
        return encoded

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

    @property
    def additional_test_metrics(self):
        return [
            self.interpolation_mse,
            self.interpolation_mae,
            # self.interpolation_relative_mse,
            # self.interpolation_relative_mae
        ]
