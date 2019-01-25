import tensorflow as tffrom keras.layers import Input, Conv2D, Reshapefrom keras.optimizers import Adamfrom keras.callbacks import CallbackListimport numpy as npfrom models import AutoEncoderBaseModel, KerasModelfrom scheme import Databasefrom generators import NoisyImagesGeneratorclass ProgAE(AutoEncoderBaseModel):    def build_model(self, config_file):        self.load_config(config_file)        self.build_layers()        self.keras_model = self.build_model_for_scale(self.depth - 1)    def build_model_for_scale(self, scale):        scale_input_shape = self.input_shape_by_scale[scale]        scale_channels = scale_input_shape[-1]        input_shape = scale_input_shape[:-1] + [self.input_channels]        with tf.name_scope("model_scale_{0}".format(scale)):            input_layer = Input(input_shape)            layer = input_layer            # Encoder            if scale is not (self.depth - 1):                layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)            for i in range(scale + 1):                layer = self.link_encoder_conv_layer(layer, scale, i)            # Embeddings            with tf.name_scope("embeddings"):                layer = Reshape([-1])(layer)                layer = self.embeddings_layer(layer)                layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)                embeddings_reshape = self.config["embeddings_reshape"]                embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)                layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)            # Decoder            for i in range(scale + 1):                layer = self.link_decoder_deconv_layer(layer, scale, i)            output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",                                  activation=self.output_activation)(layer)        model = KerasModel(inputs=input_layer, outputs=output_layer)        optimizer = Adam(lr=2e-4, decay=5e-6)        model.compile(optimizer, "mse", metrics=["mae"])        self._models_per_scale[scale] = model        return model    @property    def can_be_pre_trained(self):        return True