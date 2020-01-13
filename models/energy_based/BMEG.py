# EBGAN : Bimodal Energy-based Generative Adversarial Network
import tensorflow as tf
from tensorflow_core.python.keras import Model
from typing import Dict

from models import CustomModel


class ModalityModels(Model):
    def __init__(self,
                 encoder: Model,
                 generator: Model,
                 decoder: Model,
                 name: str,
                 **kwargs,
                 ):
        super(ModalityModels, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.generator = generator
        self.decoder = decoder

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            self.encoder: "encoder_{}".format(self.name),
            self.generator: "generator_{}".format(self.name),
            self.decoder: "decoder_{}".format(self.name),
        }

    def get_config(self):
        config = {
            "encoder": self.encoder.get_config(),
            "generator": self.generator.get_config(),
            "decoder": self.decoder.get_config(),
        }
        return config


class BMEG(CustomModel):
    def __init__(self,
                 models_1: ModalityModels,
                 models_2: ModalityModels,
                 fusion_autoencoder: Model,
                 energy_margin: float,
                 autoencoder_optimizer: tf.keras.optimizers.Optimizer,
                 generators_optimizer: tf.keras.optimizers.Optimizer,
                 **kwargs):
        super(BMEG, self).__init__(**kwargs)
        self.models_1 = models_1
        self.models_2 = models_2
        self.fusion_autoencoder = fusion_autoencoder
        self.energy_margin = energy_margin
        self.autoencoder_optimizer = autoencoder_optimizer
        self.generators_optimizer = generators_optimizer

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        losses, gradients = self.get_gradients(inputs)

        encoders_gradients, discriminator_gradients, generators_gradients = gradients
        encoders_loss, discriminator_loss, generators_loss = losses

        self.autoencoder_optimizer.apply_gradients(zip(encoders_gradients, self.encoders_trainable_variables))
        self.autoencoder_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator_trainable_variables))
        self.generators_optimizer.apply_gradients(zip(generators_gradients, self.generators_trainable_variables))

        total_loss = encoders_loss + discriminator_loss + generators_loss
        return total_loss, encoders_loss, discriminator_loss, generators_loss

    @tf.function
    def get_gradients(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as encoders_tape, \
                tf.GradientTape(watch_accessed_variables=False) as discriminator_tape, \
                tf.GradientTape(watch_accessed_variables=False) as generators_tape:
            encoders_tape.watch(self.encoders_trainable_variables)
            discriminator_tape.watch(self.discriminator_trainable_variables)
            generators_tape.watch(self.generators_trainable_variables)

            losses = self.compute_loss(inputs)
            (
                encoders_loss,
                discriminator_loss,
                generators_loss
            ) = losses

        encoders_gradient = encoders_tape.gradient(encoders_loss, self.encoders_trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator_trainable_variables)
        generators_gradient = generators_tape.gradient(generators_loss, self.generators_trainable_variables)
        gradients = (encoders_gradient, discriminator_gradient, generators_gradient)

        return losses, gradients

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        input_1, input_2 = inputs

        latent_code_1 = self.models_1.encoder(input_1)
        latent_code_2 = self.models_2.encoder(input_2)

        noise_1 = tf.random.normal(shape=tf.shape(latent_code_1))
        noise_2 = tf.random.normal(shape=tf.shape(latent_code_2))

        generator_1_input = tf.stop_gradient(latent_code_1) + noise_1
        generator_2_input = tf.stop_gradient(latent_code_2) + noise_2

        generated_1 = self.models_1.generator(generator_1_input)
        generated_2 = self.models_2.generator(generator_2_input)

        generated_1_encoded = self.models_1.encoder(generated_1)
        generated_2_encoded = self.models_2.encoder(generated_2)

        fused_1_1 = self.fusion_autoencoder([latent_code_1, latent_code_2])
        fused_1_0 = self.fusion_autoencoder([latent_code_1, generated_2_encoded])
        fused_0_1 = self.fusion_autoencoder([generated_1_encoded, latent_code_2])
        fused_0_0 = self.fusion_autoencoder([generated_1_encoded, generated_2_encoded])

        reconstructed_1_1 = self.decode(fused_1_1)
        reconstructed_1_0 = self.decode(fused_1_0)
        reconstructed_0_1 = self.decode(fused_0_1)
        reconstructed_0_0 = self.decode(fused_0_0)

        energy_1_1 = self.compute_reconstruction_energy([input_1, input_2], reconstructed_1_1)
        energy_1_0 = self.compute_reconstruction_energy([input_1, generated_2], reconstructed_1_0)
        energy_0_1 = self.compute_reconstruction_energy([generated_1, input_2], reconstructed_0_1)
        energy_0_0 = self.compute_reconstruction_energy([generated_1, generated_2], reconstructed_0_0)

        loss_1_0 = tf.nn.relu(self.energy_margin - energy_1_0)
        loss_0_1 = tf.nn.relu(self.energy_margin - energy_0_1)
        loss_0_0 = tf.nn.relu(self.energy_margin - energy_0_0) * 2.0

        encoders_loss = energy_1_1
        discriminator_loss = energy_1_1 + loss_1_0 + loss_0_1 + loss_0_0
        generators_loss = energy_1_0 + energy_0_1 + energy_0_0

        return encoders_loss, discriminator_loss, generators_loss

    @tf.function
    def autoencode(self, inputs):
        latent_codes = self.encode(inputs)
        fused = self.fusion_autoencoder(latent_codes)
        reconstructed = self.decode(fused)
        return reconstructed

    @tf.function
    def encode(self, inputs):
        input_1, input_2 = inputs
        latent_code_1 = self.models_1.encoder(input_1)
        latent_code_2 = self.models_2.encoder(input_2)
        return latent_code_1, latent_code_2

    @tf.function
    def decode(self, latent_codes):
        latent_code_1, latent_code_2 = latent_codes
        decoded_1 = self.models_1.decoder(latent_code_1)
        decoded_2 = self.models_2.decoder(latent_code_2)
        return decoded_1, decoded_2

    @tf.function
    def regenerate(self, inputs):
        latent_code_1, latent_code_2 = self.encode(inputs)

        noise_1 = tf.random.normal(shape=tf.shape(latent_code_1))
        noise_2 = tf.random.normal(shape=tf.shape(latent_code_2))

        latent_code_1 += noise_1
        latent_code_2 += noise_2

        generated = self.generate((latent_code_1, latent_code_2))
        return generated

    @tf.function
    def generate(self, latent_codes):
        latent_code_1, latent_code_2 = latent_codes
        generated_1 = self.models_1.generator(latent_code_1)
        generated_2 = self.models_2.generator(latent_code_2)
        return generated_1, generated_2

    @staticmethod
    def compute_reconstruction_energy(inputs, reconstructed):
        input_1, input_2 = inputs
        reconstructed_1, reconstructed_2 = reconstructed

        reduction_axis_1 = tuple(range(1, input_1.shape.rank))
        reduction_axis_2 = tuple(range(1, input_2.shape.rank))

        error_1 = tf.reduce_mean(tf.square(input_1 - reconstructed_1), axis=reduction_axis_1)
        error_2 = tf.reduce_mean(tf.square(input_2 - reconstructed_2), axis=reduction_axis_2)

        error = error_1 + error_2
        return error

    @property
    def encoders_trainable_variables(self):
        return self.models_1.encoder.trainable_variables + self.models_2.encoder.trainable_variables

    @property
    def discriminator_trainable_variables(self):
        decoders = self.models_1.decoder.trainable_variables + self.models_2.decoder.trainable_variables
        return decoders + self.fusion_autoencoder.trainable_variables

    @property
    def generators_trainable_variables(self):
        return self.models_1.generator.trainable_variables + self.models_2.generator.trainable_variables

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **self.models_1.models_ids,
            **self.models_2.models_ids,
            self.fusion_autoencoder: "fusion_autoencoder"
        }

    @property
    def metrics_names(self):
        return ["total_loss", "encoders_loss", "discriminator_loss", "generators_loss"]

    def get_config(self):
        config = {
            self.models_1.name: self.models_1.get_config(),
            self.models_2.name: self.models_2.get_config(),
            "fusion_autoencoder": self.fusion_autoencoder,
            "energy_margin": self.energy_margin,
            "autoencoder_optimizer": self.autoencoder_optimizer,
            "generators_optimizer": self.generators_optimizer,
        }
        return config
