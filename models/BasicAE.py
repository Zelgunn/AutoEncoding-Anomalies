from keras.layers import Input, Reshape
import tensorflow as tf

from models import AutoEncoderBaseModel, KerasModel, metrics_dict


class BasicAE(AutoEncoderBaseModel):
    def build(self):
        autoencoded = self.decoder(self.encoder(self.encoder.input))
        autoencoder = KerasModel(inputs=self.encoder.inputs, outputs=autoencoded)
        reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]]
        autoencoder.compile(self.optimizer, reconstruction_loss, metrics=self.config["metrics"])

        self._autoencoder = autoencoder

    def build_encoder(self):
        input_layer = Input(self.input_shape)
        layer = input_layer

        for i in range(self.depth):
            use_dropout = i > 0
            layer = self.encoder_layers[i](layer, use_dropout)

        # region Embeddings
        with tf.name_scope("embeddings"):
            if self.use_dense_embeddings:
                layer = Reshape([-1])(layer)
            layer = self.embeddings_layer(layer)
            layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)
            if self.use_dense_embeddings:
                layer = Reshape(self.compute_embeddings_output_shape())(layer)
        # endregion

        output_layer = layer

        self._encoder = KerasModel(inputs=input_layer, outputs=output_layer, name="Encoder")
