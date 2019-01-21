from keras.layers import Input, Conv2D, Dense, Reshape
from keras.callbacks import CallbackList
import tensorflow as tf
import numpy as np
from collections import namedtuple
from typing import List

from models import AutoEncoderBaseModel, KerasModel, metrics_dict
from layers import ResBlock2D, SpectralNormalization
from scheme import Database
from callbacks import AUCCallback

GAN_Scale = namedtuple("GAN_Scale", ["encoder", "decoder", "discriminator",
                                     "adversarial_generator", "autoencoder"])


class GAN(AutoEncoderBaseModel):
    def __init__(self,
                 image_summaries_max_outputs=3):
        super(GAN, self).__init__(image_summaries_max_outputs)
        self.discriminator_layers = []
        self._scales: List[GAN_Scale] = []

    # region Model building
    def build_layers(self):
        super(GAN, self).build_layers()

        use_res_block = ("use_resblock" in self.config["use_resblock"])
        use_res_block &= self.config["use_resblock"] == "True"
        use_spectral_norm = ("use_spectral_norm" in self.config["use_spectral_norm"])
        use_spectral_norm &= self.config["use_spectral_norm"] == "True"

        for layer_info in self.config["discriminator"]:
            if use_res_block:
                layer = ResBlock2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"])
            else:
                layer = Conv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                               padding=layer_info["padding"])

            if use_spectral_norm:
                layer = SpectralNormalization(layer)

            self.discriminator_layers.append(layer)

    def build_model(self, config_file: str):
        self.load_config(config_file)
        self._scales = [None] * self.depth
        self.build_layers()

    def build_model_for_scale(self, scale: int):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)
        discriminator = self.build_discriminator_for_scale(scale)

        scale_input_shape = self.scales_input_shapes[scale]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_input = Input(input_shape)
        autoencoded = decoder(encoder(encoder_input))
        autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoded,
                                 name="AutoEncoder_scale_{0}".format(scale))

        decoder_input = Input([self.embeddings_size])
        generator_discriminated = discriminator(decoder(decoder_input))
        adversarial_generator = KerasModel(inputs=decoder_input, outputs=generator_discriminated,
                                           name="AdversarialGenerator_scale_{0}".format(scale))

        discriminator_loss = metrics_dict[self.config["losses"]["discriminator"]]
        discriminator_metrics = self.config["metrics"]["discriminator"]
        discriminator.compile(self.optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

        autoencoder_loss = metrics_dict[self.config["losses"]["autoencoder"]]
        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        discriminator.trainable = False
        adversarial_generator_loss = metrics_dict[self.config["losses"]["generator"]]
        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.compile(self.optimizer, loss=adversarial_generator_loss,
                                      metrics=adversarial_generator_metrics)

        self._scales[scale] = GAN_Scale(encoder, decoder, discriminator, adversarial_generator, autoencoder)
        self._models_per_scale[scale] = autoencoder
        return autoencoder

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.scales_input_shapes[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_name = "Encoder_scale_{0}".format(scale)
        with tf.name_scope(encoder_name):
            input_layer = Input(input_shape)
            layer = input_layer

            if scale is not (self.depth - 1):
                layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

            for i in range(scale + 1):
                layer = self.link_encoder_conv_layer(layer, scale, i)

            with tf.name_scope("embeddings"):
                layer = Reshape([-1])(layer)
                layer = Dense(units=self.embeddings_size)(layer)
                layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)

            outputs = layer
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder

    def build_decoder_for_scale(self, scale: int):
        decoder_name = "Decoder_scale_{0}".format(scale)
        with tf.name_scope(decoder_name):
            input_layer = Input([self.embeddings_size])
            layer = input_layer

            embeddings_reshape = self.config["embeddings_reshape"]
            embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
            layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)

            for i in range(scale + 1):
                layer = self.link_decoder_deconv_layer(layer, scale, i)

            output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                                  activation="sigmoid")(layer)
        decoder = KerasModel(inputs=input_layer, outputs=output_layer, name=decoder_name)
        return decoder

    def build_discriminator_for_scale(self, scale: int):
        scale_input_shape = self.scales_input_shapes[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        discriminator_name = "Discriminator_scale_{0}".format(scale)
        with tf.name_scope(discriminator_name):
            input_layer = Input(input_shape)
            layer = input_layer

            if scale is not (self.depth - 1):
                layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

            for i in range(scale + 1):
                layer = self.link_encoder_conv_layer(layer, scale, i)

            layer = Reshape([-1])(layer)
            layer = Dense(units=1, activation="sigmoid")(layer)

            outputs = layer
        discriminator = KerasModel(inputs=input_layer, outputs=outputs, name=discriminator_name)
        return discriminator

    def get_scale_models(self, scale: int = None) -> GAN_Scale:
        if scale is None:
            scale = self.depth - 1
        if self._scales[scale] is None:
            self.build_model_for_scale(scale)
        return self._scales[scale]

    # endregion

    @property
    def can_be_pre_trained(self):
        return True

    def train_epoch(self,
                    train_generator,
                    test_generator=None,
                    scale: int = None,
                    callbacks: CallbackList = None):
        epoch_length = len(train_generator)
        scale_models = self.get_scale_models(scale)
        autoencoder: KerasModel = scale_models.autoencoder
        decoder: KerasModel = scale_models.decoder
        adversarial_generator: KerasModel = scale_models.adversarial_generator
        discriminator: KerasModel = scale_models.discriminator
        base_model = self.get_model_at_scale(scale)

        callbacks.on_epoch_begin(self.epochs_seen)
        discriminator_metrics = [0, 0.5]
        for batch_index in range(epoch_length):
            x, y = train_generator[0]
            batch_size = x.shape[0]
            z = np.random.normal(size=[batch_size, self.embeddings_size])
            zeros = np.zeros(shape=[batch_size])
            ones = np.ones(shape=[batch_size * 2])

            x_autoencoded = autoencoder.predict(x=x)
            x_generated = decoder.predict(x=z)
            x_combined = np.concatenate([y, x_autoencoded, x_generated])
            y_combined = np.concatenate([zeros, ones])
            shuffle_indices = np.random.permutation(np.arange(batch_size * 3))
            x_combined = x_combined[shuffle_indices]
            y_combined = y_combined[shuffle_indices]

            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            autoencoder_metrics = autoencoder.train_on_batch(x=x, y=y)
            generator_metrics = adversarial_generator.train_on_batch(x=z, y=zeros)
            if discriminator_metrics[1] < 0.8:
                discriminator_metrics = discriminator.train_on_batch(x=x_combined, y=y_combined)

            def add_metrics_to_batch_logs(model_name, losses):
                metric_names = ["loss", *self.config["metrics"][model_name]]
                for i in range(len(losses)):
                    batch_logs[model_name + '_' + metric_names[i]] = losses[i]

            add_metrics_to_batch_logs("autoencoder", autoencoder_metrics)
            add_metrics_to_batch_logs("generator", generator_metrics)
            add_metrics_to_batch_logs("discriminator", discriminator_metrics)

            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(base_model, train_generator, test_generator, callbacks)

    def build_anomaly_callbacks(self,
                                database: Database,
                                scale: int = None):
        scale_shape = self.scales_input_shapes[scale]
        database = database.resized_to_scale(scale_shape)
        anomaly_callbacks = super(GAN, self).build_anomaly_callbacks(database, scale)

        discriminator: KerasModel = self.get_scale_models(scale).discriminator
        discriminator_prediction = discriminator.get_output_at(0)
        discriminator_inputs_placeholder = discriminator.get_input_at(0)
        disc_auc_callback = AUCCallback(self.tensorboard, discriminator_prediction, discriminator_inputs_placeholder,
                                        database.test_dataset, plot_size=(256, 256), batch_size=128,
                                        name="Discriminator_AUC")

        anomaly_callbacks.append(disc_auc_callback)
        return anomaly_callbacks

    def callback_metrics(self, model: KerasModel):
        def model_metric_names(model_name):
            metric_names = ["loss", *self.config["metrics"][model_name]]
            return [model_name + "_" + metric_name for metric_name in metric_names]

        return model_metric_names("autoencoder") + model_metric_names("generator") + model_metric_names("discriminator")
