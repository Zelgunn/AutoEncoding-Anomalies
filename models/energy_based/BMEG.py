# EBGAN : Bimodal Energy-based Generative Adversarial Network
import tensorflow as tf
from tensorflow_core.python.keras import Model
from typing import Dict, Union, List

from models import CustomModel
from models.utils import gradient_difference_loss, reduce_mean_from


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
    def noise_size(self) -> int:
        base_code_size = self.decoder.input_shape[-1]
        noise_size = self.generator.input_shape[-1] - base_code_size
        return noise_size

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
        self.energy_margin = tf.constant(energy_margin, dtype=tf.float32, name="energy_margin")
        self.autoencoder_optimizer = autoencoder_optimizer
        self.generators_optimizer = generators_optimizer

        # self.adversarial_emphasis = tf.Variable(initial_value=0.0, dtype=tf.float32, name="adversarial_emphasis",
        #                                         trainable=False)

        self.adversarial_emphasis = self.add_weight(name="adversarial_emphasis", shape=[], dtype=tf.float32,
                                                    initializer="zeros", trainable=False)
        # self.step = self.add_weight(name="step", shape=[], dtype=tf.int32, initializer="zeros", trainable=False)

    # region Training
    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        metrics, gradients = self.get_gradients(inputs)

        encoders_gradients, discriminator_gradients, generators_gradients = gradients

        self.autoencoder_optimizer.apply_gradients(zip(encoders_gradients, self.encoders_trainable_variables))
        self.autoencoder_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator_trainable_variables))
        self.generators_optimizer.apply_gradients(zip(generators_gradients, self.generators_trainable_variables))

        return metrics

    @tf.function
    def get_gradients(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as encoders_tape, \
                tf.GradientTape(watch_accessed_variables=False) as discriminator_tape, \
                tf.GradientTape(watch_accessed_variables=False) as generators_tape:
            encoders_tape.watch(self.encoders_trainable_variables)
            discriminator_tape.watch(self.discriminator_trainable_variables)
            generators_tape.watch(self.generators_trainable_variables)

            metrics = self.compute_loss(inputs)
            losses = metrics[1:4]
            (
                encoders_loss,
                discriminator_loss,
                generators_loss
            ) = losses

            weights_decay = self.weights_decay_loss(l2=1e-5)
            encoders_loss += weights_decay
            discriminator_loss += weights_decay
            generators_loss += weights_decay

        encoders_gradient = encoders_tape.gradient(encoders_loss, self.encoders_trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator_trainable_variables)
        generators_gradient = generators_tape.gradient(generators_loss, self.generators_trainable_variables)

        gradients = (encoders_gradient, discriminator_gradient, generators_gradient)

        return metrics, gradients

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        input_1, input_2 = inputs

        latent_codes = self.encode(inputs)
        latent_code_1, latent_code_2 = latent_codes

        generated_1, generated_2 = self.generate(latent_codes)

        generated_encoded = self.encode([generated_1, generated_2])
        generated_1_encoded, generated_2_encoded = generated_encoded

        fused_1_1 = self.fusion_autoencoder(latent_codes)
        fused_1_0 = self.fusion_autoencoder([latent_code_1, generated_2_encoded])
        fused_0_1 = self.fusion_autoencoder([generated_1_encoded, latent_code_2])
        fused_0_0 = self.fusion_autoencoder(generated_encoded)

        reconstructed_1_1 = self.decode(fused_1_1)
        reconstructed_1_0 = self.decode(fused_1_0)
        reconstructed_0_1 = self.decode(fused_0_1)
        reconstructed_0_0 = self.decode(fused_0_0)

        energy_1_1 = self.compute_reconstruction_energy(inputs, reconstructed_1_1)
        energy_1_0 = self.compute_reconstruction_energy([input_1, generated_2], reconstructed_1_0)
        energy_0_1 = self.compute_reconstruction_energy([generated_1, input_2], reconstructed_0_1)
        energy_0_0 = self.compute_reconstruction_energy([generated_1, generated_2], reconstructed_0_0)
        generators_energy = tf.reduce_mean((energy_1_0, energy_0_1, energy_0_0), axis=0)

        pull_away_loss = self.batch_pull_away_loss(generated_encoded)
        pull_away_factor = pull_away_loss
        # pull_away_loss = tf.nn.relu(pull_away_loss - 0.5)
        latent_code_error = self.latent_code_distance_loss(latent_codes, generated_encoded)
        simple_autoencoded = self.decode(latent_codes)
        # simple_autoencode_loss = self.compute_reconstruction_energy(inputs, simple_autoencoded)

        encoders_loss = energy_1_1  # + simple_autoencode_loss * 0.5
        discriminator_loss = energy_1_1 - generators_energy * self.adversarial_emphasis
        generators_loss = generators_energy + latent_code_error + pull_away_loss * 0.5
        # generators_loss = generators_energy + pull_away_loss * 0.5

        # region Update Emphasis rate & Convergence measure
        emphasis_rate = tf.constant(1e-4, dtype=tf.float32, name="emphasis_rate")
        diversity_ratio = tf.constant(0.7, dtype=tf.float32, name="diversity_ratio")
        balance = diversity_ratio * energy_1_1 - generators_energy

        adversarial_emphasis = tf.reduce_mean(emphasis_rate * balance)
        adversarial_emphasis = tf.clip_by_value(self.adversarial_emphasis + adversarial_emphasis, 0.0, 1.0)
        self.adversarial_emphasis.assign(adversarial_emphasis)

        convergence_measure = energy_1_1 + tf.abs(balance)
        # endregion

        total_loss = encoders_loss + discriminator_loss + generators_loss

        return total_loss, encoders_loss, discriminator_loss, generators_loss, convergence_measure, pull_away_factor

    # endregion

    # region Encode/Decode/Generate
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
        latent_codes = self.encode(inputs)
        generated = self.generate(latent_codes)
        return generated

    # @tf.function
    def generate(self, latent_codes):
        latent_code_1, latent_code_2 = latent_codes
        batch_size = tf.shape(latent_code_1)[0]
        noise = self.get_noise(batch_size)
        return self.generate_with_given_noise(latent_codes, noise)

    @tf.function
    def get_noise(self, batch_size, name="noise"):
        return tf.random.normal(shape=[batch_size, self.noise_size], name=name)

    @tf.function
    def generate_with_given_noise(self, latent_codes, noise):
        latent_code_1, latent_code_2 = latent_codes
        generated_1 = self.models_1.generator([latent_code_2, noise])
        generated_2 = self.models_2.generator([latent_code_1, noise])
        return generated_1, generated_2

    # endregion

    # region Losses
    @staticmethod
    def compute_reconstruction_energy(inputs, reconstructed) -> tf.Tensor:
        input_1, input_2 = inputs
        reconstructed_1, reconstructed_2 = reconstructed

        error_1 = reduce_mean_from(tf.square(input_1 - reconstructed_1))
        error_2 = reduce_mean_from(tf.square(input_2 - reconstructed_2))

        # gradient_axis_1 = tuple(range(1, input_1.shape.rank - 1))
        # gradient_axis_2 = tuple(range(1, input_2.shape.rank - 1))

        # gradient_error_1 = reduce_mean_from(gradient_difference_loss(input_1, reconstructed_1, axis=gradient_axis_1))
        # gradient_error_2 = reduce_mean_from(gradient_difference_loss(input_2, reconstructed_2, axis=gradient_axis_2))

        # true_contrast = tf.math.reduce_variance(input_2, axis=(2, 3, 4))
        # recon_contrast = tf.math.reduce_variance(input_2, axis=(2, 3, 4))
        # contrast_error = reduce_mean_from(tf.abs(true_contrast - recon_contrast))

        error = error_1 + error_2
        # error = error + (gradient_error_1 + gradient_error_2) * 0.5
        # error = error + contrast_error
        return error

    @staticmethod
    def batch_pull_away_loss(latent_code: Union[List[tf.Tensor], tf.Tensor]):
        if isinstance(latent_code, (tuple, list)):
            losses = [BMEG.batch_pull_away_loss(code) for code in latent_code]
            return tf.reduce_mean(losses)

        if latent_code.shape.rank > 2:
            batch_size, *dimensions = tf.unstack(tf.shape(latent_code))
            new_shape = [batch_size, tf.reduce_prod(dimensions)]
            latent_code = tf.reshape(latent_code, new_shape)
        else:
            batch_size = tf.shape(latent_code)[0]

        latent_code_norm = tf.norm(latent_code, ord=2, axis=-1, keepdims=True)
        latent_code = latent_code / latent_code_norm

        similarity = latent_code @ tf.transpose(latent_code)

        batch_size = tf.cast(batch_size, similarity.dtype)
        loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return loss

    @staticmethod
    def latent_code_distance_loss(true_encoded, pred_encoded):
        if isinstance(true_encoded, (tuple, list)):
            losses = [BMEG.latent_code_distance_loss(a, b) for a, b in zip(true_encoded, pred_encoded)]
            return tf.reduce_mean(losses)

        # reduction_axis = tuple(range(1, true_encoded.shape.rank))
        # distance = tf.losses.cosine_similarity(true_encoded, pred_encoded, axis=reduction_axis)
        # distance = (1.0 + distance) * 0.5
        distance = reduce_mean_from(tf.square(true_encoded - pred_encoded))
        return distance

    # endregion

    # region Trainable variables
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

    # endregion

    @property
    def noise_size(self) -> int:
        return self.models_1.generator.input_shape[1][-1]

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **self.models_1.models_ids,
            **self.models_2.models_ids,
            self.fusion_autoencoder: "fusion_autoencoder"
        }

    @property
    def metrics_names(self):
        return ["total_loss", "encoders_loss", "discriminator_loss", "generators_loss", "convergence", "pull_away"]

    def get_config(self):
        config = {
            self.models_1.name: self.models_1.get_config(),
            self.models_2.name: self.models_2.get_config(),
            "fusion_autoencoder": self.fusion_autoencoder,
            "energy_margin": self.energy_margin.numpy(),
            "autoencoder_optimizer": self.autoencoder_optimizer,
            "generators_optimizer": self.generators_optimizer,
        }
        return config
