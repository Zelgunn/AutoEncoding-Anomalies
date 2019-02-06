from keras.layers import Input, Conv2D, Reshape
import numpy as np

from models import AutoEncoderBaseModel, VariationalBaseModel, VAEScale, KerasModel, metrics_dict
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class VAE(VariationalBaseModel):
    def build(self, config_file: str):
        self.load_config(config_file)
        self.build_layers()

    def build_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        input_layer = Input(input_shape)
        layer = input_layer

        # region Encoder
        if scale is not (self.depth - 1):
            layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)
        # endregion

        # region Embeddings
        layer = Reshape([-1])(layer)

        latent_mean = self.latent_mean_layer(layer)
        latent_log_var = self.latent_log_var_layer(layer)
        layer = self.embeddings_layer([latent_mean, latent_log_var])

        layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)
        embeddings_reshape = self.config["embeddings_reshape"]
        embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
        layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)
        # endregion

        # region Decoder
        for i in range(scale + 1):
            layer = self.link_decoder_deconv_layer(layer, scale, i)
        # endregion

        output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                              activation=self.output_activation)(layer)

        def vae_loss(y_true, y_pred):
            reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]](y_true, y_pred)
            divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
            return reconstruction_loss + divergence

        autoencoder = KerasModel(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(self.optimizer, loss=vae_loss, metrics=self.config["metrics"])
        self._scales[scale] = VAEScale(encoder, decoder, autoencoder)
