from keras.layers import Input, Reshape
import tensorflow as tf

from models import GAN, VariationalBaseModel, KerasModel, metrics_dict
from models.VAE import kullback_leibler_divergence_mean0_var1


class VAEGAN(GAN, VariationalBaseModel):
    # region Model building
    def compile_encoder(self):
        input_layer = Input(self.input_shape)
        layer = input_layer

        for i in range(self.depth):
            use_dropout = i > 0
            layer = self.encoder_layers[i](layer, use_dropout)

        # region Embeddings
        with tf.name_scope("embeddings"):
            if self.use_dense_embeddings:
                layer = Reshape([-1])(layer)

            latent_mean = self.latent_mean_layer(layer)
            latent_log_var = self.latent_log_var_layer(layer)

            if not self.use_dense_embeddings:
                latent_mean = Reshape([-1])(latent_mean)
                latent_log_var = Reshape([-1])(latent_log_var)

            layer = self.embeddings_layer([latent_mean, latent_log_var])
            layer = GAN.get_activation(self.embeddings_activation)(layer)
            layer = Reshape(self.compute_embeddings_output_shape())(layer)
        # endregion

        outputs = [layer, latent_mean, latent_log_var]
        self._latent_mean = latent_mean
        self._latent_log_var = latent_log_var
        self._encoder = KerasModel(inputs=input_layer, outputs=outputs, name="Encoder")

    def compile(self):
        with tf.name_scope("Autoencoder"):
            encoder_input = Input(self.input_shape)
            encoded, latent_mean, latent_log_var = self.encoder(encoder_input)
            autoencoded = self.decoder(encoded)
            autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoded,
                                     name="Autoencoder")

        with tf.name_scope("Adversarial_Generator"):
            decoder_input = Input(self.compute_decoder_input_shape())
            generator_discriminated = self.discriminator(self.decoder(decoder_input))
            adversarial_generator = KerasModel(inputs=decoder_input, outputs=generator_discriminated,
                                               name="Adversarial_Generator")

        with tf.name_scope("Training"):
            with tf.name_scope("Discriminator"):
                discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]

                def discriminator_loss(y_true, y_pred):
                    return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

                discriminator_metrics = self.config["metrics"]["discriminator"]
                self.discriminator.compile(self.optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

            with tf.name_scope("Autoencoder"):
                with tf.name_scope("autoencoder_loss"):
                    reconstruction_metric = metrics_dict[self.config["losses"]["autoencoder"]]
                    divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
                    divergence *= self.config["loss_weights"]["divergence"]

                    def autoencoder_loss(y_true, y_pred):
                        reconstruction_loss = reconstruction_metric(y_true, y_pred)
                        reconstruction_loss *= self.config["loss_weights"]["reconstruction"]
                        return reconstruction_loss + divergence

                autoencoder_metrics = self.config["metrics"]["autoencoder"]
                autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

            with tf.name_scope("Adversarial_Generator"):
                self.discriminator.trainable = False
                adversarial_generator_metrics = self.config["metrics"]["generator"]
                adversarial_generator.compile(self.optimizer, loss=discriminator_loss,
                                              metrics=adversarial_generator_metrics)

        self._autoencoder = autoencoder
        self._adversarial_generator = adversarial_generator
    # endregion
