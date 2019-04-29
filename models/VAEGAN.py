from tensorflow.python.keras.models import Model as KerasModel
import tensorflow as tf

from models import GAN, VariationalBaseModel, metrics_dict
from models.VAE import kullback_leibler_divergence_mean0_var1


class VAEGAN(GAN, VariationalBaseModel):
    # region Compile
    def compile_encoder(self):
        VariationalBaseModel.compile_encoder(self)

    def compile(self):
        # region KerasModel (init)
        with tf.name_scope("Autoencoder"):
            encoded, latent_mean, latent_log_var = self.encoder(self.encoder_input_layer)
            autoencoded = self.decoder(encoded)
            autoencoder = KerasModel(inputs=self.encoder_input_layer, outputs=autoencoded,
                                     name="Autoencoder")

        # region Adversarial generator (train generator with discriminator)
        with tf.name_scope("Adversarial_Generator"):
            decoder_trainable = KerasModel(inputs=self.decoder.input,
                                           outputs=self.decoder.output,
                                           name="decoder_trainable")
            decoder_trainable.trainable = True

            discriminator_non_trainable = KerasModel(inputs=self.discriminator.input,
                                                     outputs=self.discriminator.output,
                                                     name="discriminator_non_trainable")
            discriminator_non_trainable.trainable = False

            adversarial_generator_output = discriminator_non_trainable(decoder_trainable(self.decoder_input_layer))
            adversarial_generator = KerasModel(inputs=self.decoder_input_layer,
                                               outputs=adversarial_generator_output,
                                               name="Adversarial_Generator")
        # endregion

        # region Adversarial discriminator (train discriminator with fake data)
        with tf.name_scope("Adversarial_Discriminator"):
            decoder_non_trainable = KerasModel(inputs=self.decoder.input,
                                               outputs=self.decoder.output,
                                               name="decoder_non_trainable")
            decoder_non_trainable.trainable = False

            discriminator_trainable = KerasModel(inputs=self.discriminator.input,
                                                 outputs=self.discriminator.output,
                                                 name="discriminator_trainable")
            discriminator_trainable.trainable = True
            adversarial_discriminator_output = discriminator_trainable(decoder_non_trainable(self.decoder_input_layer))
            adversarial_discriminator = KerasModel(inputs=self.decoder_input_layer,
                                                   outputs=adversarial_discriminator_output,
                                                   name="Adversarial_Discriminator")
        # endregion
        # endregion

        with tf.name_scope("Training"):
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
                adversarial_generator_metrics = self.config["metrics"]["generator"]
                adversarial_generator.add_loss(self.discriminator_loss_real_data(adversarial_generator_output))
                adversarial_generator.compile(self.optimizer,
                                              metrics=adversarial_generator_metrics)

            with tf.name_scope("Adversarial_Discriminator"):
                adversarial_discriminator_metrics = self.config["metrics"]["discriminator"]
                adversarial_discriminator.add_loss(self.discriminator_loss_fake_data(adversarial_discriminator_output))
                adversarial_discriminator.compile(self.optimizer,
                                                  metrics=adversarial_discriminator_metrics)

        self._autoencoder = autoencoder
        self._adversarial_generator = adversarial_generator
        self._adversarial_discriminator = adversarial_discriminator
    # endregion
