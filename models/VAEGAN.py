from keras.layers import Dense, Lambda, Input, Conv2D, Reshape
import tensorflow as tf
import numpy as np

from models import GAN, KerasModel, metrics_dict
from models.VAE import sampling, kullback_leibler_divergence_mean0_var1
from models.GAN import GAN_Scale


class VAEGAN(GAN):
    # region Initialization
    def __init__(self):
        super(VAEGAN, self).__init__()
        self.latent_mean = None
        self.latent_log_var = None
        self.kl_divergence_by_scale = None

    def load_config(self, config_file: str):
        super(VAEGAN, self).load_config(config_file)
        self.kl_divergence_by_scale = [None] * self.depth

    # endregion

    # region Model building
    def build_layers(self):
        super(VAEGAN, self).build_layers()
        self.latent_mean = self.embeddings_layer
        if self.use_dense_embeddings:
            self.latent_log_var = Dense(units=self.embeddings_size,
                                        kernel_regularizer=self.weight_decay_regularizer,
                                        bias_regularizer=self.weight_decay_regularizer)
        else:
            self.latent_log_var = Conv2D(filters=self.embeddings_size, kernel_size=3, padding="same",
                                         kernel_regularizer=self.weight_decay_regularizer,
                                         bias_regularizer=self.weight_decay_regularizer)

        self.embeddings_layer = Lambda(function=sampling)

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_name = "Encoder_scale_{0}".format(scale)
        with tf.name_scope(encoder_name):
            input_layer = Input(input_shape)
            layer = input_layer

            if scale is not (self.depth - 1):
                layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

            for i in range(scale + 1):
                layer = self.link_encoder_conv_layer(layer, scale, i)

            with tf.name_scope("embeddings"):
                if self.use_dense_embeddings:
                    layer = Reshape([-1])(layer)

                latent_mean = self.latent_mean(layer)
                latent_log_var = self.latent_log_var(layer)

                if not self.use_dense_embeddings:
                    latent_mean = Reshape([-1])(latent_mean)
                    latent_log_var = Reshape([-1])(latent_log_var)

                layer = self.embeddings_layer([latent_mean, latent_log_var])

                layer = GAN.get_activation(self.embeddings_activation)(layer)
                embeddings_reshape = self.config["embeddings_reshape"]
                embeddings_filters = self.embeddings_size
                if self.use_dense_embeddings:
                    embeddings_filters = embeddings_filters // np.prod(embeddings_reshape)
                layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)

            outputs = [layer, latent_mean, latent_log_var]
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder

    def build_model_for_scale(self, scale: int):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)
        discriminator = self.build_discriminator_for_scale(scale)

        scale_input_shape = self.input_shape_by_scale[scale]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_input = Input(input_shape)
        encoded, latent_mean, latent_log_var = encoder(encoder_input)
        autoencoded = decoder(encoded)
        autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoded,
                                 name="AutoEncoder_scale_{0}".format(scale))

        decoder_input = Input(self.embeddings_shape)
        generator_discriminated = discriminator(decoder(decoder_input))
        adversarial_generator = KerasModel(inputs=decoder_input, outputs=generator_discriminated,
                                           name="AdversarialGenerator_scale_{0}".format(scale))

        discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]

        def discriminator_loss(y_true, y_pred):
            return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

        discriminator_metrics = self.config["metrics"]["discriminator"]
        discriminator.compile(self.optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

        reconstruction_metric = metrics_dict[self.config["losses"]["autoencoder"]]
        divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
        divergence *= self.config["loss_weights"]["divergence"]

        def autoencoder_loss(y_true, y_pred):
            reconstruction_loss = reconstruction_metric(y_true, y_pred) * self.config["loss_weights"]["reconstruction"]
            return reconstruction_loss + divergence

        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        discriminator.trainable = False
        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.compile(self.optimizer, loss=discriminator_loss, metrics=adversarial_generator_metrics)

        self._scales[scale] = GAN_Scale(encoder, decoder, discriminator, adversarial_generator, autoencoder)
        self._models_per_scale[scale] = autoencoder
        return autoencoder
    # endregion
