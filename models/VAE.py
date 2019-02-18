from keras.layers import Input, Reshape
import tensorflow as tf

from models import AutoEncoderBaseModel, VariationalBaseModel, VAEScale, KerasModel, metrics_dict
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class VAE(VariationalBaseModel):
    def build_for_scale(self, scale: int):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)

        input_layer = encoder.input
        encoded, latent_mean, latent_log_var = encoder(input_layer)
        autoencoded = decoder(encoded)

        def vae_loss(y_true, y_pred):
            reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]](y_true, y_pred)
            divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
            return reconstruction_loss + divergence

        autoencoder = KerasModel(inputs=input_layer, outputs=autoencoded,
                                 name="AutoEncoder_scale_{0}".format(scale))
        autoencoder.compile(self.optimizer, loss=vae_loss, metrics=self.config["metrics"])
        self._scales[scale] = VAEScale(encoder=encoder, decoder=decoder, autoencoder=autoencoder,
                                       latent_mean=latent_mean, latent_log_var=latent_log_var)

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.channels_count]

        encoder_name = "Encoder_scale_{0}".format(scale)
        input_layer = Input(input_shape)
        layer = input_layer

        # region Encoder
        if scale is not (self.depth - 1):
            layer = self.build_adaptor_layer(scale_channels, self.encoder_rank)(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)
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
            layer = Reshape(self.embeddings_shape)(layer)
        # endregion

        outputs = [layer, latent_mean, latent_log_var]
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder
