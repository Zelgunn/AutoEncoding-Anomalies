from keras.layers import Input, Conv2D, Dense, Deconv2D, Reshape, Dropout
import numpy as np

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel


class BasicAE(AutoEncoderBaseModel):
    def build(self, config_file):
        self.load_config(config_file)

        default_activation = self.config["default_activation"]
        embeddings_activation = self.config["embeddings_activation"]

        # region Encoder
        encoder_input = Input(self.input_shape)
        layer = encoder_input

        i = 1
        for layer_info in self.config["encoder"]:
            layer = Conv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                           padding=layer_info["padding"])(layer)

            layer = AutoEncoderBaseModel.get_activation(default_activation)(layer)
            if "dropout" in layer_info:
                layer = Dropout(rate=layer_info["dropout"], name="encoder_dropout_{0}".format(i))(layer)
            i += 1

        layer = Reshape([-1])(layer)
        layer = Dense(units=self.config["embeddings_size"])(layer)
        layer = AutoEncoderBaseModel.get_activation(embeddings_activation)(layer)
        embeddings_reshape = self.config["embeddings_reshape"]
        embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
        layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)
        encoder_output = layer
        # endregion

        # region Decoder
        decoder_input = Input(self.embeddings_shape)
        layer = decoder_input

        i = 1
        for layer_info in self.config["decoder"]:
            layer = Deconv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                             padding=layer_info["padding"], name="deconv2d_{0}".format(i))(layer)

            layer = AutoEncoderBaseModel.get_activation(default_activation)(layer)
            if "dropout" in layer_info:
                layer = Dropout(rate=layer_info["dropout"], name="decoder_dropout_{0}".format(i))(layer)
            i += 1

        decoder_output = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                                activation=self.output_activation)(layer)
        # endregion

        encoder = KerasModel(inputs=encoder_input, outputs=encoder_output, name="Encoder")
        decoder = KerasModel(inputs=decoder_input, outputs=decoder_output, name="Decoder")

        autoencoder_output = decoder(encoder_output)
        autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoder_output)
        autoencoder.compile(self.optimizer, "mse", metrics=["mae"])

        self._scales[self.depth - 1] = AutoEncoderScale(encoder, decoder, autoencoder)

    def build_for_scale(self, scale):
        raise NotImplementedError

    def build_encoder_for_scale(self, scale: int):
        raise NotImplementedError

    def build_decoder_for_scale(self, scale: int):
        raise NotImplementedError
