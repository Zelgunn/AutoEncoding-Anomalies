from keras.layers import Input, Conv2D, Reshape
import numpy as np

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel, metrics_dict


class BasicAE(AutoEncoderBaseModel):
    def build(self, config_file):
        self.load_config(config_file)
        self.build_layers()

    def build_for_scale(self, scale):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)

        autoencoded = decoder(encoder(encoder.input))
        autoencoder = KerasModel(inputs=encoder.inputs, outputs=autoencoded)
        reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]]
        autoencoder.compile(self.optimizer, reconstruction_loss, metrics=self.config["metrics"])
        self._scales[scale] = AutoEncoderScale(encoder, decoder, autoencoder)

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]
        input_layer = Input(input_shape)
        layer = input_layer

        if scale is not (self.depth - 1):
            layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)

        # region Embeddings
        if self.use_dense_embeddings:
            layer = Reshape([-1])(layer)
        layer = self.embeddings_layer(layer)
        layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)
        if self.use_dense_embeddings:
            embeddings_reshape = self.config["embeddings_reshape"]
            embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
            layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)
        # endregion

        output_layer = layer

        encoder = KerasModel(inputs=input_layer, outputs=output_layer, name="Encoder-sc{0}".format(scale))
        return encoder

    def build_decoder_for_scale(self, scale: int):
        input_layer = Input(self.embeddings_shape)
        layer = input_layer

        for i in range(scale + 1):
            layer = self.link_decoder_deconv_layer(layer, scale, i)

        output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                              activation=self.output_activation)(layer)

        decoder = KerasModel(inputs=input_layer, outputs=output_layer, name="Decoder-sc{0}".format(scale))
        return decoder
