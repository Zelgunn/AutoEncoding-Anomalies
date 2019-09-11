import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import Model
from enum import IntEnum
from typing import Tuple, Dict

from models import VAE


class GANLoss(IntEnum):
    BASE = 0,
    LEAST_SQUARE = 1,
    WASSERSTEIN = 2


class GANLossTarget(IntEnum):
    GENERATOR_FAKE = 0,
    DISCRIMINATOR_FAKE = 1,
    DISCRIMINATOR_REAL = 2


class VAEGAN(VAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 autoencoder_learning_rate=1e-3,
                 discriminator_learning_rate=1e-4,
                 balance_discriminator_learning_rate=True,
                 reconstruction_loss_factor=100.0,
                 learned_reconstruction_loss_factor=1000.0,
                 kl_divergence_loss_factor=1.0,
                 loss_used=GANLoss.WASSERSTEIN,
                 **kwargs):
        super(VAEGAN, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        self.autoencoder_learning_rate = autoencoder_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.balance_discriminator_learning_rate = balance_discriminator_learning_rate

        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.learned_reconstruction_loss_factor = learned_reconstruction_loss_factor
        self.kl_divergence_loss_factor = kl_divergence_loss_factor
        self.loss_used = loss_used

        self.generator_fake_loss = tf.Variable(initial_value=1.0)
        self.discriminator_fake_loss = tf.Variable(initial_value=1.0)

        self.encoder_optimizer = tf.keras.optimizers.Adam(autoencoder_learning_rate, beta_1=0.5, beta_2=0.9)
        self.decoder_optimizer = tf.keras.optimizers.Adam(autoencoder_learning_rate, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self._get_discriminator_learning_rate,
                                                                beta_1=0.5, beta_2=0.999)

    def call(self, inputs, training=None, mask=None):
        latent_distribution = self.get_latent_distribution(inputs)
        z = latent_distribution.sample()
        decoded = self.decoder(z)
        return decoded

    # region Training
    @property
    def metrics_names(self):
        return ["reconstruction", "learned_reconstruction", "kl_divergence",
                "decoder_fake", "generator_fake",
                "discriminator_real", "discriminator_fake"]

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as encoder_tape, \
                tf.GradientTape() as decoder_tape, \
                tf.GradientTape() as discriminator_tape:
            losses = self.compute_loss(inputs)
            (
                reconstruction_loss,
                learned_reconstruction_loss,
                kl_divergence,
                decoder_fake_loss,
                generator_fake_loss,
                discriminator_real_loss,
                discriminator_fake_loss
            ) = losses

            reconstruction_loss *= self.reconstruction_loss_factor
            learned_reconstruction_loss *= self.learned_reconstruction_loss_factor
            kl_divergence *= self.kl_divergence_loss_factor

            encoder_loss = reconstruction_loss + kl_divergence + learned_reconstruction_loss + decoder_fake_loss
            decoder_loss = reconstruction_loss + learned_reconstruction_loss + generator_fake_loss + decoder_fake_loss
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss

        encoder_gradients = encoder_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        decoder_gradients = decoder_tape.gradient(decoder_loss, self.decoder.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator.trainable_variables)

        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.decoder_optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        if self.loss_used == GANLoss.WASSERSTEIN:
            [variable.assign(tf.clip_by_value(variable, -0.001, 0.001))
             for variable in self.discriminator.trainable_variables]

        return losses

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        latent_distribution = self.get_latent_distribution(inputs)
        z = latent_distribution.sample()
        decoded = self.decoder(z)

        noise = tf.random.normal(shape=z.shape, mean=0.0, stddev=1.0)
        generated = self.decoder(noise)

        discriminated_inputs, discriminated_inputs_early = self.discriminator(inputs)
        discriminated_decoded, discriminated_decoded_early = self.discriminator(decoded)
        discriminated_generated, discriminated_generated_early = self.discriminator(generated)

        reconstruction_loss = tf.reduce_mean(tf.square(decoded - inputs))

        learned_reconstruction_loss = tf.square(discriminated_inputs_early - discriminated_decoded_early)
        learned_reconstruction_loss = tf.reduce_mean(learned_reconstruction_loss)
        learned_reconstruction_loss = learned_reconstruction_loss

        reference_distribution = get_reference_distribution(latent_distribution)
        kl_divergence = tfp.distributions.kl_divergence(latent_distribution, reference_distribution)
        kl_divergence = tf.maximum(kl_divergence, 0.0)
        kl_divergence = tf.reduce_mean(kl_divergence)

        decoder_fake_loss = gan_loss(discriminated_decoded, self.loss_used, GANLossTarget.GENERATOR_FAKE)
        generator_fake_loss = gan_loss(discriminated_generated, self.loss_used, GANLossTarget.GENERATOR_FAKE)
        discriminator_real_loss = gan_loss(discriminated_inputs, self.loss_used, GANLossTarget.DISCRIMINATOR_REAL)
        discriminator_fake_loss = gan_loss(discriminated_generated, self.loss_used, GANLossTarget.DISCRIMINATOR_FAKE)

        # if self.loss_used == GANLoss.WASSERSTEIN:
        #     discriminator_gradients = tf.gradients(discriminator_fake_loss, generated)
        #     gradient_penalty = tf.square(discriminator_gradients)
        #     reduce_axis = [0] + list(range(2, gradient_penalty.shape.rank))
        #     gradient_penalty = tf.reduce_sum(gradient_penalty, axis=reduce_axis)
        #     gradient_penalty = tf.sqrt(gradient_penalty)
        #     gradient_penalty = tf.square(gradient_penalty - 1.0)
        #     gradient_penalty = tf.reduce_mean(gradient_penalty) * 10.0
        #     discriminator_fake_loss += gradient_penalty

        if self.balance_discriminator_learning_rate:
            update_losses = [self.generator_fake_loss.assign(generator_fake_loss),
                             self.discriminator_fake_loss.assign(discriminator_fake_loss)]
            with tf.control_dependencies(update_losses):
                generator_fake_loss = tf.identity(generator_fake_loss)

        return (
            reconstruction_loss,
            learned_reconstruction_loss,
            kl_divergence,
            decoder_fake_loss,
            generator_fake_loss,
            discriminator_real_loss,
            discriminator_fake_loss
        )

    # endregion

    def _get_discriminator_learning_rate(self):
        if not self.balance_discriminator_learning_rate:
            return self.discriminator_learning_rate

        loss_difference = self.discriminator_fake_loss - self.generator_fake_loss
        ratio = tf.nn.sigmoid(loss_difference)
        return self.discriminator_learning_rate * ratio

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "discriminator": self.discriminator.get_config(),
            "autoencoder_learning_rate": self.autoencoder_learning_rate,
            "discriminator_learning_rate": self.discriminator_learning_rate,
            "balance_discriminator_learning_rate": self.balance_discriminator_learning_rate,
            "reconstruction_loss_factor": self.reconstruction_loss_factor,
            "learned_reconstruction_loss_factor": self.learned_reconstruction_loss_factor,
            "kl_divergence_loss_factor": self.kl_divergence_loss_factor,
            "loss_used": self.loss_used
        }
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.encoder: "encoder",
                self.decoder: "decoder",
                self.discriminator: "discriminator"}


def get_reference_distribution(latent_distribution: tfp.distributions.MultivariateNormalDiag
                               ) -> tfp.distributions.MultivariateNormalDiag:
    event_size = latent_distribution.event_shape[-1]
    mean = tf.zeros(shape=[event_size])
    variance = tf.ones(shape=[event_size])
    return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=variance)


# region Losses
def gan_loss(logits: tf.Tensor, loss_used: GANLoss, target: GANLossTarget) -> tf.Tensor:
    if loss_used == GANLoss.BASE:
        return base_gan_loss(logits, is_real=target is not GANLossTarget.DISCRIMINATOR_FAKE)
    elif loss_used == GANLoss.LEAST_SQUARE:
        targets = {GANLossTarget.DISCRIMINATOR_FAKE: -1,
                   GANLossTarget.DISCRIMINATOR_REAL: 1,
                   GANLossTarget.GENERATOR_FAKE: 0}
        return least_square_gan_loss(logits, target=targets[target])
    elif loss_used == GANLoss.WASSERSTEIN:
        return wasserstein_gan_loss(logits, is_real=target is not GANLossTarget.DISCRIMINATOR_FAKE)
    else:
        raise ValueError


@tf.function
def base_gan_loss(logits: tf.Tensor, is_real: bool) -> tf.Tensor:
    labels = tf.ones_like(logits) if is_real else tf.zeros_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def least_square_gan_loss(logits: tf.Tensor, target) -> tf.Tensor:
    loss = tf.square(logits - target)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def wasserstein_gan_loss(logits: tf.Tensor, is_real: bool) -> tf.Tensor:
    factor = -1.0 if is_real else 1.0
    return tf.reduce_mean(logits) * factor
# endregion
