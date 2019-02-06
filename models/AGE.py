from keras.layers import Input, Conv2D, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import CallbackList
from keras.utils.generic_utils import to_list
import tensorflow as tf
import numpy as np
import copy
from typing import List

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel, metrics_dict
from models.VAE import kullback_leibler_divergence_mean0_var1
from datasets import Database


class AGEScale(AutoEncoderScale):
    def __init__(self,
                 encoder: KerasModel,
                 decoder: KerasModel,
                 encoder_real_data_trainer: KerasModel,
                 encoder_fake_data_trainer: KerasModel,
                 decoder_real_data_trainer: KerasModel,
                 decoder_fake_data_trainer: KerasModel,
                 **kwargs):
        super(AGEScale, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       autoencoder=encoder_real_data_trainer,
                                       **kwargs)
        self.encoder_real_data_trainer = encoder_real_data_trainer
        self.encoder_fake_data_trainer = encoder_fake_data_trainer
        self.decoder_real_data_trainer = decoder_real_data_trainer
        self.decoder_fake_data_trainer = decoder_fake_data_trainer


class AGE(AutoEncoderBaseModel):
    def __init__(self):
        super(AGE, self).__init__()
        self._scales: List[AGEScale] = []

    # region Model building
    def build(self, config_file: str):
        self.load_config(config_file)
        self._scales = [None] * self.depth
        self.build_layers()

    def build_for_scale(self, scale: int):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)

        scale_input_shape = self.input_shape_by_scale[scale]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_decoder_input = Input(input_shape)
        real_data_latent = encoder(encoder_decoder_input)
        encoder_decoder_output = decoder(real_data_latent)

        real_data_encoder_loss = self.build_loss(True, True, real_data_latent)
        real_data_decoder_loss = self.build_loss(False, True, real_data_latent)

        decoder_encoder_input = Input([self.embeddings_size])
        fake_data_latent = encoder(decoder(decoder_encoder_input))
        decoder_encoder_output = fake_data_latent

        fake_data_encoder_loss = self.build_loss(True, False, fake_data_latent)
        fake_data_decoder_loss = self.build_loss(False, False, fake_data_latent)

        # region Real Data
        encoder.trainable = True
        decoder.trainable = True
        encoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Encoder_real_data_Model_scale_{0}".format(scale))
        encoder_real_data_trainer.compile(self.optimizer, loss=real_data_encoder_loss)
        encoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Encoder_fake_data_Model_scale_{0}".format(scale))
        encoder_fake_data_trainer.compile(self.optimizer, loss=fake_data_encoder_loss)
        # endregion

        # region Fake Data
        encoder.trainable = False
        decoder.trainable = True
        decoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Decoder_real_data_Model_scale_{0}".format(scale))
        decoder_real_data_trainer.compile(self.optimizer, loss=real_data_decoder_loss)
        decoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Decoder_fake_data_Model_scale_{0}".format(scale))
        decoder_fake_data_trainer.compile(self.optimizer, loss=fake_data_decoder_loss)
        # endregion

        scale_models = AGEScale(encoder, decoder,
                                encoder_real_data_trainer, encoder_fake_data_trainer,
                                decoder_real_data_trainer, decoder_fake_data_trainer)
        self._scales[scale] = scale_models

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        encoder_name = "Encoder_scale_{0}".format(scale)
        input_layer = Input(input_shape)
        layer = input_layer

        if scale is not (self.depth - 1):
            layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)

        latent = Reshape([-1])(layer)
        latent = Dense(units=self.embeddings_size, name="Latent_Value")(latent)
        latent = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(latent)

        outputs = latent
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder

    def build_decoder_for_scale(self, scale: int):
        decoder_name = "Decoder_scale_{0}".format(scale)
        input_layer = Input([self.embeddings_size])
        layer = input_layer

        embeddings_reshape = self.config["embeddings_reshape"]
        embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
        layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)

        for i in range(scale + 1):
            layer = self.link_decoder_deconv_layer(layer, scale, i)

        output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                              activation=self.output_activation)(layer)
        decoder = KerasModel(inputs=input_layer, outputs=output_layer, name=decoder_name)
        return decoder

    def build_loss(self, is_encoder: bool, real_data: bool, latent: tf.Tensor):
        loss_weights = self.config["loss_weights"][
            "encoder" if is_encoder else "decoder"][
            "real_data" if real_data else "fake_data"]
        reconstruction_weight: float = loss_weights["reconstruction"]
        divergence_weight: float = loss_weights["divergence"]
        reconstruction_metric_name = self.config["reconstruction_metrics"]["real" if real_data else "fake"]
        reconstruction_metric = metrics_dict[reconstruction_metric_name]

        def loss_function(y_true, y_pred):
            if reconstruction_weight != 0.0:
                axis = [1, 2, 3] if real_data else -1
                reconstruction_loss = reconstruction_metric(y_true, y_pred, axis)
                reconstruction_loss *= tf.constant(reconstruction_weight)
            else:
                reconstruction_loss: tf.Tensor = None

            if divergence_weight != 0.0:
                latent_mean = tf.reduce_mean(latent, axis=0)
                latent_var = tf.square(latent - latent_mean)
                latent_var = tf.reduce_mean(latent_var, axis=0)
                divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_var, use_variance_log=False)

                divergence *= tf.constant(divergence_weight)
            else:
                divergence: tf.Tensor = None

            if reconstruction_loss is None:
                return divergence
            elif divergence is None:
                return reconstruction_loss
            else:
                return tf.reduce_mean(reconstruction_loss) + divergence

        return loss_function

    def get_scale_models(self, scale: int = None) -> AGEScale:
        if scale is None:
            scale = self.depth - 1
        if self._scales[scale] is None:
            self.build_for_scale(scale)
        return self._scales[scale]

    # endregion

    # region Training
    @property
    def can_be_pre_trained(self):
        return True

    def train_epoch(self,
                    database: Database,
                    scale: int = None,
                    callbacks: CallbackList = None):
        epoch_length = len(database.train_dataset)
        scale_models = self.get_scale_models(scale)
        base_model = self.get_autoencoder_model_at_scale(scale)

        callbacks.on_epoch_begin(self.epochs_seen)

        for batch_index in range(epoch_length):
            decoder_steps = self.config["decoder_steps"]
            x, y = database.train_dataset[0]
            batch_size = x.shape[0]
            z = np.random.normal(size=[decoder_steps + 1, batch_size, self.embeddings_size])

            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            # Train Encoder
            encoder_real_data_loss = scale_models.encoder_real_data_trainer.train_on_batch(x=x, y=y)
            encoder_fake_data_loss = scale_models.encoder_fake_data_trainer.train_on_batch(x=z[0], y=z[0])

            # Train Decoder
            decoder_fake_data_losses = []
            for i in range(1, decoder_steps + 1):
                decoder_fake_data_loss = scale_models.decoder_fake_data_trainer.train_on_batch(x=z[i], y=z[i])
                decoder_fake_data_losses.append(decoder_fake_data_loss)
            decoder_fake_data_loss = float(np.mean(decoder_fake_data_losses))

            batch_logs["loss"] = encoder_real_data_loss
            batch_logs["fake_loss"] = encoder_fake_data_loss
            batch_logs["decoder_loss"] = decoder_fake_data_loss
            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(base_model, database, callbacks)

    def on_epoch_end(self,
                     base_model: KerasModel,
                     train_generator,
                     test_generator=None,
                     callbacks: CallbackList = None,
                     epoch_logs: dict = None):
        if epoch_logs is None:
            epoch_logs = {}

        if test_generator:
            out_labels = base_model.metrics_names
            val_outs = base_model.evaluate_generator(test_generator)
            val_outs = to_list(val_outs)
            for label, val_out in zip(out_labels, val_outs):
                epoch_logs["val_{0}".format(label)] = val_out
            test_generator.on_epoch_end()

        train_generator.on_epoch_end()
        if callbacks:
            callbacks.on_epoch_end(self.epochs_seen, epoch_logs)
        self.epochs_seen += 1

    # endregion

    # region Callbacks (building)
    def callback_metrics(self, model: KerasModel):
        out_labels = model.metrics_names
        real_data_metrics = copy.copy(out_labels)
        fake_data_metrics = ["fake_{0}".format(name) for name in out_labels]
        decoder_metrics = ["decoder_{0}".format(name) for name in out_labels]
        val_metrics = ["val_{0}".format(name) for name in out_labels]
        return real_data_metrics + fake_data_metrics + decoder_metrics + val_metrics
    # endregion
