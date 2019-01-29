import tensorflow as tffrom keras.callbacks import CallbackListfrom keras.layers import Input, Conv2D, Dense, Deconv2D, Reshape, Dropoutfrom keras.optimizers import Adamimport numpy as npfrom models import AutoEncoderBaseModel, KerasModelfrom datasets import Databaseclass BasicAE(AutoEncoderBaseModel):    def build_model(self, config_file):        self.load_config(config_file)        default_activation = self.config["default_activation"]        embeddings_activation = self.config["embeddings_activation"]        input_layer = Input(self.input_shape)        layer = input_layer        with tf.name_scope("encoder"):            i = 1            for layer_info in self.config["encoder"]:                layer = Conv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],                               padding=layer_info["padding"])(layer)                layer = AutoEncoderBaseModel.get_activation(default_activation)(layer)                if "dropout" in layer_info:                    layer = Dropout(rate=layer_info["dropout"], name="encoder_dropout_{0}".format(i))(layer)                i += 1        with tf.name_scope("embeddings"):            layer = Reshape([-1])(layer)            layer = Dense(units=self.config["embeddings_size"])(layer)            layer = AutoEncoderBaseModel.get_activation(embeddings_activation)(layer)            embeddings_reshape = self.config["embeddings_reshape"]            embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)            layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)        with tf.name_scope("decoder"):            i = 1            for layer_info in self.config["decoder"]:                layer = Deconv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],                                 padding=layer_info["padding"], name="deconv2d_{0}".format(i))(layer)                layer = AutoEncoderBaseModel.get_activation(default_activation)(layer)                if "dropout" in layer_info:                    layer = Dropout(rate=layer_info["dropout"], name="decoder_dropout_{0}".format(i))(layer)                i += 1            output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",                                  activation=self.output_activation)(layer)        self.keras_model = KerasModel(inputs=input_layer, outputs=output_layer)        optimizer = Adam(lr=2e-4, decay=5e-6)        self.keras_model.compile(optimizer, "mse", metrics=["mae"])        self._models_per_scale = [self.keras_model]    def build_model_for_scale(self, scale):        pass    def get_scales_input_shapes(self):        return [self.input_shape]    def pre_train_scale(self, database: Database, callbacks: CallbackList, scale: int, batch_size, epoch_length, epochs,                        **kwargs):        pass    def train_loop(self, database: Database, callbacks, batch_size, epoch_length, epochs, scale, **kwargs):        pass