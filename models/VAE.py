from keras.layers import Input, Reshape
import tensorflow as tf

from models import AutoEncoderBaseModel, VariationalBaseModel, KerasModel, metrics_dict
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class VAE(VariationalBaseModel):
    def compile(self):
        input_layer = self.encoder.input
        encoded, latent_mean, latent_log_var = self.encoder(input_layer)
        autoencoded = self.decoder(encoded)

        def vae_loss(y_true, y_pred):
            reconstruction_loss_function = self.get_reconstruction_loss(self.config["reconstruction_loss"])
            reconstruction_loss = reconstruction_loss_function(y_true, y_pred)
            divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
            return reconstruction_loss + divergence * 0.001

        autoencoder = KerasModel(inputs=input_layer, outputs=autoencoded, name="AutoEncoder")
        autoencoder.compile(self.optimizer, loss=vae_loss, metrics=self.config["metrics"])

        self._autoencoder = autoencoder
        self._latent_mean = latent_mean
        self._latent_log_var = latent_log_var

    def compile_encoder(self):
        input_layer = Input(self.input_shape)
        layer = input_layer

        # region Encoder
        for i in range(self.depth):
            use_dropout = i > 0
            layer = self.encoder_layers[i](layer, use_dropout)
        # endregion

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
            layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)
            layer = Reshape(self.compute_embeddings_output_shape())(layer)
        # endregion

        outputs = [layer, latent_mean, latent_log_var]
        self._encoder = KerasModel(inputs=input_layer, outputs=outputs, name="Encoder")
