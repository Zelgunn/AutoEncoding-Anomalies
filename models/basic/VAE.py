# VAE : Variational Autoencoder
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import Model
from typing import Tuple

from models import AE


class VAE(AE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 learning_rate=1e-3,
                 reconstruction_loss_factor=100.0,
                 kl_divergence_loss_factor=1.0,
                 **kwargs):
        super(VAE, self).__init__(encoder=encoder,
                                  decoder=decoder,
                                  learning_rate=learning_rate,
                                  **kwargs)
        self.kl_divergence_loss_factor = kl_divergence_loss_factor
        self.reconstruction_loss_factor = reconstruction_loss_factor
        self.training_step = tf.Variable(initial_value=0, trainable=False, name="training_step")

    @tf.function
    def encode(self, inputs):
        return self.sample_latent_distribution(inputs)

    def get_latent_distribution(self, inputs) -> tfp.distributions.MultivariateNormalDiag:
        encoder_outputs = self.encoder(inputs)
        latent_mean, latent_variance = tf.split(encoder_outputs, num_or_size_splits=2, axis=-1)
        latent_distribution = tfp.distributions.MultivariateNormalDiag(loc=latent_mean, scale_diag=latent_variance)
        return latent_distribution

    @tf.function
    def sample_latent_distribution(self, inputs) -> tf.Tensor:
        return self.get_latent_distribution(inputs).sample()

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_divergence = self.compute_loss(inputs)
            loss = reconstruction_loss + kl_divergence

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        reconstruction_loss /= self.reconstruction_loss_factor
        kl_divergence /= self.cyclic_kl_divergence_factor()

        self.training_step.assign_add(1)
        return reconstruction_loss, kl_divergence

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor, tf.Tensor]:
        latent_distribution = self.get_latent_distribution(inputs)
        reference_distribution = get_reference_distribution(latent_distribution)
        latent_code = latent_distribution.sample()
        decoded = self.decode(latent_code)

        reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))
        reconstruction_loss *= self.reconstruction_loss_factor
        kl_divergence = tf.reduce_mean(tfp.distributions.kl_divergence(latent_distribution, reference_distribution))
        kl_divergence *= self.cyclic_kl_divergence_factor()

        return reconstruction_loss, kl_divergence

    @tf.function
    def cyclic_kl_divergence_factor(self):
        step_size = 1000
        step = tf.math.mod(self.training_step, step_size * 2)
        if step < step_size:
            factor = self.kl_divergence_loss_factor * tf.cast(step, tf.float32) / step_size
        else:
            factor = self.kl_divergence_loss_factor
        return factor

    def get_config(self):
        return {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "learning_rate": self.learning_rate,
            "kl_divergence_factor": self.kl_divergence_loss_factor
        }

    @property
    def metrics_names(self):
        return ["reconstruction", "kl_divergence"]


def get_reference_distribution(latent_distribution: tfp.distributions.MultivariateNormalDiag
                               ) -> tfp.distributions.MultivariateNormalDiag:
    event_size = latent_distribution.event_shape[-1]
    mean = tf.zeros(shape=[event_size])
    variance = tf.ones(shape=[event_size])
    return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=variance)
