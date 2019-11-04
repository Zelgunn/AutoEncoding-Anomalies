import tensorflow as tf
from tensorflow.python.keras import Model
from typing import Tuple, Dict
from enum import IntEnum

from models import IAE
from models.adversarial import gan_loss


class IAEGANMode(IntEnum):
    ENCODED_VS_INTERPOLATED = 0,
    NORMAL_VS_ENCODED = 1,
    INPUTS_VS_OUTPUTS = 2


class IAEGAN(IAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 step_size: int,
                 mode: IAEGANMode,
                 autoencoder_learning_rate=1e-3,
                 discriminator_learning_rate=1e-3,
                 ):
        super(IAEGAN, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     step_size=step_size,
                                     learning_rate=autoencoder_learning_rate)
        self.mode = mode
        self.discriminator = discriminator
        self.discriminator_learning_rate = discriminator_learning_rate
        # self.discriminator.optimizer = tf.keras.optimizers.SGD(learning_rate=self.discriminator_learning_rate)
        self.discriminator.optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_learning_rate,
                                                                beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as encoder_tape, \
                tf.GradientTape() as decoder_tape, \
                tf.GradientTape() as discriminator_tape:
            losses = self.compute_loss(inputs)
            (
                reconstruction_loss,
                adversarial_loss,
                discriminator_fake_loss,
                discriminator_real_loss
            ) = losses

            # adversarial_loss_factor = tf.constant(1e-2)
            reconstruction_loss_factor = tf.constant(1e2)

            encoder_loss = reconstruction_loss * reconstruction_loss_factor
            decoder_loss = reconstruction_loss * reconstruction_loss_factor + adversarial_loss
            discriminator_loss = (discriminator_fake_loss + discriminator_real_loss)

        encoder_gradients = encoder_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        decoder_gradients = decoder_tape.gradient(decoder_loss, self.decoder.trainable_variables)
        disc_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return losses

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.mode == IAEGANMode.ENCODED_VS_INTERPOLATED:
            return self.compute_encoded_vs_interpolated_loss(inputs)
        elif self.mode == IAEGANMode.NORMAL_VS_ENCODED:
            return self.compute_normal_vs_encoded_loss(inputs)
        elif self.mode == IAEGANMode.INPUTS_VS_OUTPUTS:
            return self.compute_inputs_vs_outputs_loss(inputs)
        else:
            raise AttributeError("Mode is not valid : {}".format(self.mode))

    # region Losses (1 per mode)
    @tf.function
    def compute_encoded_vs_interpolated_loss(self, inputs):
        inputs_shape = tf.shape(inputs)

        latent_code = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=False)
        base_code = tf.stack([latent_code[:, 0], latent_code[:, -1]], axis=1)
        interpolated_code = self.select_two_latent_codes(latent_code)
        latent_code_shape = tf.shape(latent_code)
        batch_size, step_count, *sample_code_shape = tf.unstack(latent_code_shape)

        # region Reconstruction loss
        latent_code = tf.reshape(latent_code, [batch_size * step_count, *sample_code_shape])
        decoded = self.decode(latent_code)
        decoded = tf.reshape(decoded, inputs_shape)
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
        # endregion

        # region Adversarial loss
        base_code = tf.reshape(base_code, [batch_size * 2, *sample_code_shape])
        interpolated_code = tf.reshape(interpolated_code, [batch_size * 2, *sample_code_shape])

        adversarial_losses = self.compute_adversarial_losses(base_code, interpolated_code)
        encoder_adversarial_loss, discriminator_fake_loss, discriminator_real_loss = adversarial_losses
        # endregion

        return reconstruction_loss, encoder_adversarial_loss, discriminator_fake_loss, discriminator_real_loss

    @tf.function
    def compute_normal_vs_encoded_loss(self, inputs):
        inputs_shape = tf.shape(inputs)

        latent_code = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        latent_code_shape = tf.shape(latent_code)

        # region Reconstruction loss
        decoded = self.decode(latent_code)
        decoded = tf.reshape(decoded, inputs_shape)
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
        # endregion

        # region Adversarial loss
        normal_code = tf.random.normal(latent_code_shape)
        adversarial_losses = self.compute_adversarial_losses(normal_code, latent_code)
        encoder_adversarial_loss, discriminator_fake_loss, discriminator_real_loss = adversarial_losses
        # endregion

        return reconstruction_loss, encoder_adversarial_loss, discriminator_fake_loss, discriminator_real_loss

    @tf.function
    def compute_inputs_vs_outputs_loss(self, inputs):
        latent_code = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        outputs = self.decode(latent_code)
        inputs = tf.reshape(inputs, tf.shape(outputs))

        reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
        adversarial_losses = self.compute_adversarial_losses(inputs, outputs)
        generator_adversarial_loss, discriminator_fake_loss, discriminator_real_loss = adversarial_losses

        return reconstruction_loss, generator_adversarial_loss, discriminator_fake_loss, discriminator_real_loss

    @tf.function
    def compute_adversarial_losses(self, real_inputs, fake_inputs):
        real_discriminated = self.discriminator(real_inputs)
        fake_discriminated = self.discriminator(fake_inputs)

        generator_adversarial_loss = gan_loss(fake_discriminated, is_real=True)
        discriminator_fake_loss = gan_loss(fake_discriminated, is_real=False)
        discriminator_real_loss = gan_loss(real_discriminated, is_real=True)

        return generator_adversarial_loss, discriminator_fake_loss, discriminator_real_loss

    # region select_two_latent_codes (from interpolated latent code)
    @tf.function
    def select_two_latent_codes(self, latent_code):
        step_count = tf.shape(latent_code)[1]

        def select_center():
            return latent_code[:, 1:-1]

        def select_random():
            return self.select_two_random_latent_codes(latent_code)

        selected = tf.cond(step_count == 4,
                           true_fn=select_center,
                           false_fn=select_random)

        return selected

    @tf.function
    def select_two_random_latent_codes(self, latent_code):
        code_shape = tf.shape(latent_code)
        batch_size = code_shape[0]
        step_count = code_shape[1]

        batch_range = tf.range(batch_size, dtype=tf.int32)
        batch_range = tf.reshape(batch_range, [batch_size, 1, 1])
        batch_range = tf.tile(batch_range, [1, 2, 1])

        first_indices = tf.random.uniform([batch_size], minval=1, maxval=step_count - 1, dtype=tf.int32)
        second_indices = tf.random.uniform([batch_size], minval=1, maxval=step_count - 1, dtype=tf.int32)
        second_indices = self.select_next_if_same(first_indices, second_indices, step_count)

        indices = tf.stack([first_indices, second_indices], axis=-1)
        indices = tf.expand_dims(indices, axis=-1)
        batch_indices = tf.concat([batch_range, indices], axis=-1)

        return tf.gather_nd(latent_code, batch_indices)

    @tf.function
    def select_next_if_same(self, first_indices, second_indices, step_count):
        def elem_compare(indices):
            def move_if_last():
                return tf.cond(indices[0] == step_count - 2,
                               true_fn=lambda: 1,
                               false_fn=lambda: indices[0] + 1)

            return tf.cond(indices[0] == indices[1],
                           true_fn=move_if_last,
                           false_fn=lambda: indices[1])

        return tf.map_fn(fn=elem_compare, elems=tf.stack([first_indices, second_indices], axis=-1))

    # endregion
    # endregion

    # region Additional test metrics
    @property
    def additional_test_metrics(self):
        metrics = [
            *super(IAEGAN, self).additional_test_metrics,
            self.discriminate,
        ]

        if self.mode == IAEGANMode.ENCODED_VS_INTERPOLATED:
            metrics.append(self.discriminate_encoded_vs_interpolated)

        return metrics

    @tf.function
    def discriminate(self, inputs):
        if self.mode == IAEGANMode.INPUTS_VS_OUTPUTS:
            batch_size = tf.shape(inputs)[0]
            inputs = self.split_inputs(inputs, merge_batch_and_steps=True)
            result = self.discriminator(inputs)
            result = tf.reshape(result, [batch_size, -1])
            return result
        else:
            return self.discriminate_latent_code(inputs, only_interpolated=False)

    @tf.function
    def discriminate_encoded_vs_interpolated(self, inputs):
        return self.discriminate_latent_code(inputs, only_interpolated=True)

    def discriminate_latent_code(self, inputs, only_interpolated):
        latent_code = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=False)

        if only_interpolated:
            latent_code = latent_code[:, 1: -1]

        batch_size, step_count, *dimensions = tf.unstack(tf.shape(latent_code))
        latent_code = tf.reshape(latent_code, [batch_size * step_count, *dimensions])

        discriminated = self.discriminator(latent_code)
        discriminated = tf.reshape(discriminated, [batch_size, step_count])

        return discriminated

    # endregion

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **super(IAEGAN, self).models_ids,
            self.discriminator: "discriminator"
        }

    @property
    def metrics_names(self):
        return ["reconstruction", "generator_adversarial", "discriminator_fake", "discriminator_real"]
