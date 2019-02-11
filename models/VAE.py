from keras.layers import Input, Conv2D, Reshape
import numpy as np

from models import AutoEncoderBaseModel, VariationalBaseModel, VAEScale, KerasModel, metrics_dict
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class VAE(VariationalBaseModel):
    def build(self, config_file: str):
        self.load_config(config_file)
        self.build_layers()

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
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_name = "Encoder_scale_{0}".format(scale)
        input_layer = Input(input_shape)
        layer = input_layer

        # region Encoder
        if scale is not (self.depth - 1):
            layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)
        # endregion

        # region Embeddings
        if self.use_dense_embeddings:
            layer = Reshape([-1])(layer)

        latent_mean = self.latent_mean_layer(layer)
        latent_log_var = self.latent_log_var_layer(layer)

        if not self.use_dense_embeddings:
            latent_mean = Reshape([-1])(latent_mean)
            latent_log_var = Reshape([-1])(latent_log_var)

        layer = self.embeddings_layer([latent_mean, latent_log_var])

        layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)

        embeddings_reshape = self.config["embeddings_reshape"]
        embeddings_filters = self.embeddings_size
        if self.use_dense_embeddings:
            embeddings_filters = embeddings_filters // np.prod(embeddings_reshape)
        layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)
        # endregion

        outputs = [layer, latent_mean, latent_log_var]
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder

    def build_decoder_for_scale(self, scale: int):
        decoder_name = "Decoder_scale_{0}".format(scale)
        input_layer = Input(self.embeddings_shape)
        layer = input_layer

        for i in range(scale + 1):
            layer = self.link_decoder_deconv_layer(layer, scale, i)

        output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                              activation=self.output_activation)(layer)

        decoder = KerasModel(inputs=input_layer, outputs=output_layer, name=decoder_name)
        return decoder
