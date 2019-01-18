from keras.layers import Input, Conv2D, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import CallbackList, ProgbarLogger, BaseLogger, LearningRateScheduler
from keras.utils.generic_utils import to_list
import tensorflow as tf
import numpy as np
import copy
from collections import namedtuple
from typing import List

from models import AutoEncoderBaseModel, KerasModel
from scheme import Database
from generators import NoisyImagesGenerator

AGE_Scale = namedtuple("AGE_Scale", ["encoder", "decoder",
                                     "encoder_real_data_trainer", "encoder_fake_data_trainer",
                                     "decoder_real_data_trainer", "decoder_fake_data_trainer"])


class AGE(AutoEncoderBaseModel):
    def __init__(self,
                 image_summaries_max_outputs: int):
        super(AGE, self).__init__(image_summaries_max_outputs)
        self._scales: List[AGE_Scale] = []

    # region Model building
    def build_model(self, config_file):
        self.load_config(config_file)
        self._scales = [None] * self.depth
        self.build_layers()

    def build_model_for_scale(self, scale):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)

        scale_input_shape = self.scales_input_shapes[scale]
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

        optimizer = Adam(lr=self.config["optimizer"]["lr"],
                         beta_1=self.config["optimizer"]["beta_1"],
                         beta_2=self.config["optimizer"]["beta_2"],
                         decay=self.config["optimizer"]["decay"])

        # Real Data
        encoder.trainable = True
        decoder.trainable = True
        encoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Encoder_real_data_Model_scale_{0}".format(scale))
        encoder_real_data_trainer.compile(optimizer, loss=real_data_encoder_loss)
        encoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Encoder_fake_data_Model_scale_{0}".format(scale))
        encoder_fake_data_trainer.compile(optimizer, loss=fake_data_encoder_loss)

        # Fake Data
        encoder.trainable = False
        decoder.trainable = True
        decoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Decoder_real_data_Model_scale_{0}".format(scale))
        decoder_real_data_trainer.compile(optimizer, loss=real_data_decoder_loss)
        decoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Decoder_fake_data_Model_scale_{0}".format(scale))
        decoder_fake_data_trainer.compile(optimizer, loss=fake_data_decoder_loss)

        scale_models = AGE_Scale(encoder, decoder,
                                 encoder_real_data_trainer, encoder_fake_data_trainer,
                                 decoder_real_data_trainer, decoder_fake_data_trainer)
        self._scales[scale] = scale_models

        # Base model
        encoder.trainable = True
        decoder.trainable = True

        base_model = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                name="AGE_scale_{0}".format(scale))
        base_model.compile(optimizer=Adam(lr=1e-4), loss="mse", metrics=["mae"])
        self._models_per_scale[scale] = base_model
        return base_model

    def build_encoder_for_scale(self, scale):
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
                latent = Reshape([-1])(layer)
                latent = Dense(units=self.embeddings_size, name="Latent_Value")(latent)
                latent = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(latent)

            outputs = latent
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name="encoder_name")
        return encoder

    def build_decoder_for_scale(self, scale):
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

    def build_loss(self, is_encoder: bool, real_data: bool, latent):
        loss_weights = self.config["loss_weights"][
            "encoder" if is_encoder else "decoder"][
            "real_data" if real_data else "fake_data"]
        reconstruction_weight: float = loss_weights["reconstruction"]
        divergence_weight: float = loss_weights["divergence"]
        reconstruction_metric_name = self.config["reconstruction_metrics"]["real" if real_data else "fake"]
        reconstruction_metric = reconstruction_metrics[reconstruction_metric_name]

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
                divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_var)

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

    def get_scale_models(self, scale: int = None) -> AGE_Scale:
        if scale is None:
            scale = self.depth - 1
        if self._scales[scale] is None:
            self.build_model_for_scale(scale)
        return self._scales[scale]

    # endregion

    # region Training
    @property
    def can_be_pre_trained(self):
        return True

    def pre_train_scale(self, database: Database, callbacks: CallbackList, scale: int, batch_size, epoch_length, epochs,
                        **kwargs):
        self.train_loop(database, callbacks, batch_size, epoch_length, epochs, scale)
        # scale_shape = self.scales_input_shapes[scale]
        # database = database.resized_to_scale(scale_shape)
        # train_images = database.train_dataset.images
        # test_images = database.test_dataset.images
        #
        # train_generator = NoisyImagesGenerator(train_images, dropout_rate=0.0, batch_size=batch_size,
        #                                        epoch_length=epoch_length)
        # test_generator = NoisyImagesGenerator(test_images, dropout_rate=0.0, batch_size=batch_size)
        #
        # for epoch in range(epochs):
        #     self.train_epoch(epoch, train_generator, test_generator, scale=scale, callbacks=callbacks)

    def train_loop(self, database: Database, callbacks: CallbackList, batch_size, epoch_length, epochs, scale,
                   **kwargs):
        scale_shape = self.scales_input_shapes[scale]
        database = database.resized_to_scale(scale_shape)

        train_generator = NoisyImagesGenerator(database.train_dataset.images, dropout_rate=0.0, batch_size=batch_size,
                                               epoch_length=epoch_length)
        test_generator = NoisyImagesGenerator(database.test_dataset.images, dropout_rate=0.0, batch_size=batch_size)

        for _ in range(epochs):
            self.train_epoch(train_generator, test_generator, scale, callbacks)

    def train_epoch(self,
                    train_generator,
                    test_generator=None,
                    scale: int = None,
                    callbacks: CallbackList = None):
        epoch_length = len(train_generator)
        scale_models = self.get_scale_models(scale)
        base_model = self.get_model_at_scale(scale)

        self.on_epoch_begin(callbacks)

        for batch_index in range(epoch_length):
            decoder_steps = self.config["decoder_steps"]
            x, y = train_generator[0]
            batch_size = x.shape[0]
            z = np.random.normal(size=[decoder_steps + 1, train_generator.batch_size, self.embeddings_size])

            AGE.on_batch_begin(batch_index, batch_size, callbacks)

            # Train Encoder
            encoder_real_data_loss = scale_models.encoder_real_data_trainer.train_on_batch(x=x, y=y)
            encoder_fake_data_loss = scale_models.encoder_fake_data_trainer.train_on_batch(x=z[0], y=z[0])

            # Train Decoder
            decoder_fake_data_losses = []
            for i in range(1, decoder_steps + 1):
                decoder_fake_data_loss = scale_models.decoder_fake_data_trainer.train_on_batch(x=z[i], y=z[i])
                decoder_fake_data_losses.append(decoder_fake_data_loss)
            decoder_fake_data_loss = np.mean(decoder_fake_data_losses)

            AGE.on_batch_end(batch_index, batch_size,
                             encoder_real_data_loss, encoder_fake_data_loss, decoder_fake_data_loss, callbacks)

        self.on_epoch_end(base_model, train_generator, test_generator, callbacks)

    # endregion

    # region Callbacks (on_batch_x, on_epoch_x)
    @staticmethod
    def on_batch_begin(batch_index: int, batch_size: int,
                       callbacks: CallbackList = None):
        if callbacks:
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

    @staticmethod
    def on_batch_end(batch_index: int, batch_size: int,
                     encoder_real_data_loss, encoder_fake_data_loss, decoder_fake_data_loss,
                     callbacks: CallbackList = None):
        batch_logs = {"batch": batch_index, "size": batch_size,
                      "loss": encoder_real_data_loss,
                      "fake_loss": encoder_fake_data_loss,
                      "decoder_loss": decoder_fake_data_loss}
        if callbacks:
            callbacks.on_batch_end(batch_index, batch_logs)

    def on_epoch_begin(self, callbacks: CallbackList = None):
        if callbacks:
            callbacks.on_epoch_begin(self.epochs_seen)

    def on_epoch_end(self,
                     base_model,
                     train_generator, test_generator=None,
                     callbacks: CallbackList = None,
                     epoch_logs=None):
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
    def build_common_callbacks(self):
        common_callbacks = super(AGE, self).build_common_callbacks()
        stateful_metrics = []  # ["val_loss", "val_mean_absolute_error"]

        base_logger = BaseLogger(stateful_metrics=stateful_metrics)
        common_callbacks.insert(0, base_logger)

        progbar_logger = ProgbarLogger(count_mode="steps", stateful_metrics=stateful_metrics)
        common_callbacks.append(progbar_logger)

        if ("lr_drop_epochs" in self.config) and (self.config["lr_drop_epochs"] > 0):
            lr_scheduler = LearningRateScheduler(self.get_learning_rate_schedule())
            common_callbacks.append(lr_scheduler)
        return common_callbacks

    def get_learning_rate_schedule(self):
        lr_drop_epochs = self.config["lr_drop_epochs"]

        def schedule(epoch, learning_rate):
            if (epoch % lr_drop_epochs) == (lr_drop_epochs - 1):
                return learning_rate * 0.5
            else:
                return learning_rate

        return schedule

    def callback_metrics(self, model):
        out_labels = model.metrics_names
        real_data_metrics = copy.copy(out_labels)
        fake_data_metrics = ["fake_{0}".format(name) for name in out_labels]
        decoder_metrics = ["decoder_{0}".format(name) for name in out_labels]
        val_metrics = ["val_{0}".format(name) for name in out_labels]
        return real_data_metrics + fake_data_metrics + decoder_metrics + val_metrics
    # endregion


# region Utils
def sampling(args):
    latent_mean, latent_log_var = args

    shape = tf.shape(latent_mean)
    batch_size = shape[0]
    latent_dim = shape[1]

    epsilon = tf.random_normal(shape=[batch_size, latent_dim], mean=0., stddev=1.0)
    return latent_mean + tf.exp(0.5 * latent_log_var) * epsilon


def kullback_leibler_divergence_mean0_var1(mean, variance):
    divergence = (variance + tf.square(mean)) - tf.log(variance)
    divergence = (tf.reduce_mean(divergence) - 1.0) * 0.5
    return divergence
    # return 0.5 * tf.reduce_mean(-(log_variance + 1) + tf.exp(log_variance) + tf.square(mean))


def absolute_error(y_true, y_pred, axis):
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(error, axis)


def squared_error(y_true, y_pred, axis):
    error = tf.square(y_true, y_pred)
    return tf.reduce_mean(error, axis)


def cosine_distance(y_true, y_pred, axis):
    with tf.name_scope("Cosine_distance"):
        y_true = tf.nn.l2_normalize(y_true, axis=axis)
        y_pred = tf.nn.l2_normalize(y_pred, axis=axis)
        distance = 2.0 - y_true * y_pred
        return tf.reduce_mean(distance, axis)


reconstruction_metrics = {"L1": absolute_error, "L2": squared_error, "cos": cosine_distance}
# endregion
