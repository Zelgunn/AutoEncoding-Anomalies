from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Reshape
from tensorflow.python.keras.callbacks import CallbackList
import tensorflow as tf
import numpy as np
import copy
from typing import Optional

from models import AutoEncoderBaseModel, metrics_dict
from models.VAE import kullback_leibler_divergence_mean0_var1
from datasets import DatasetLoader


class AGE(AutoEncoderBaseModel):
    def __init__(self):
        super(AGE, self).__init__()

        self._encoder_real_data_trainer: Optional[KerasModel] = None
        self._encoder_fake_data_trainer: Optional[KerasModel] = None
        self._decoder_real_data_trainer: Optional[KerasModel] = None
        self._decoder_fake_data_trainer: Optional[KerasModel] = None

    # region Compile
    def compile(self):
        encoder_decoder_input = Input(self.input_shape)
        real_data_latent = self.encoder(encoder_decoder_input)
        encoder_decoder_output = self.decoder(real_data_latent)

        real_data_encoder_loss = self.build_loss(True, True, real_data_latent)
        real_data_decoder_loss = self.build_loss(False, True, real_data_latent)

        decoder_encoder_input = self.decoder_input_layer
        fake_data_latent = self.encoder(self.reconstructor_output)
        decoder_encoder_output = fake_data_latent

        fake_data_encoder_loss = self.build_loss(True, False, fake_data_latent)
        fake_data_decoder_loss = self.build_loss(False, False, fake_data_latent)

        # region Real Data
        self.encoder.trainable = True
        self.decoder.trainable = True
        encoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Encoder_real_data_Model")
        encoder_real_data_trainer.compile(self.optimizer, loss=real_data_encoder_loss)
        encoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Encoder_fake_data_Model")
        encoder_fake_data_trainer.compile(self.optimizer, loss=fake_data_encoder_loss)
        # endregion

        # region Fake Data
        self.encoder.trainable = False
        self.decoder.trainable = True
        decoder_real_data_trainer = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output,
                                               name="Decoder_real_data_Model")
        decoder_real_data_trainer.compile(self.optimizer, loss=real_data_decoder_loss)
        decoder_fake_data_trainer = KerasModel(inputs=decoder_encoder_input, outputs=decoder_encoder_output,
                                               name="Decoder_fake_data_Model")
        decoder_fake_data_trainer.compile(self.optimizer, loss=fake_data_decoder_loss)
        # endregion

        self._encoder_real_data_trainer = encoder_real_data_trainer
        self._encoder_fake_data_trainer = encoder_fake_data_trainer
        self._decoder_real_data_trainer = decoder_real_data_trainer
        self._decoder_fake_data_trainer = decoder_fake_data_trainer

        autoencoder = KerasModel(inputs=encoder_decoder_input, outputs=encoder_decoder_output, name="Autoencoder")
        self._autoencoder = autoencoder

    def compile_encoder(self):
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

        outputs = layer
        self._encoder = KerasModel(inputs=input_layer, outputs=outputs, name="Encoder")

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
                reconstruction_loss: Optional[tf.Tensor] = None

            if divergence_weight != 0.0:
                latent_mean = tf.reduce_mean(latent, axis=0)
                latent_var = tf.square(latent - latent_mean)
                latent_var = tf.reduce_mean(latent_var, axis=0)
                divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_var, use_variance_log=False)

                divergence *= tf.constant(divergence_weight)
            else:
                divergence: Optional[tf.Tensor] = None

            if reconstruction_loss is None:
                return divergence
            elif divergence is None:
                return reconstruction_loss
            else:
                return tf.reduce_mean(reconstruction_loss) + divergence

        return loss_function

    # endregion

    @property
    def encoder_real_data_trainer(self):
        if self._encoder_real_data_trainer is None:
            self.compile()
        return self._encoder_real_data_trainer

    @property
    def encoder_fake_data_trainer(self):
        if self._encoder_fake_data_trainer is None:
            self.compile()
        return self._encoder_fake_data_trainer

    @property
    def decoder_real_data_trainer(self):
        if self._decoder_real_data_trainer is None:
            self.compile()
        return self._decoder_real_data_trainer

    @property
    def decoder_fake_data_trainer(self):
        if self._decoder_fake_data_trainer is None:
            self.compile()
        return self._decoder_fake_data_trainer

    # region Training
    def train_epoch(self,
                    dataset: DatasetLoader,
                    callbacks: CallbackList,
                    batch_size: int,
                    epoch_length: int
                    ):
        callbacks.on_epoch_begin(self.epochs_seen)
        dataset_iterator = dataset.train_subset.tf_dataset.batch(batch_size)

        for batch_index in range(epoch_length):
            decoder_steps = self.config["decoder_steps"]
            z = np.random.normal(size=[decoder_steps + 1, batch_size, self.embeddings_size])

            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            # Train Encoder
            encoder_real_data_loss = self._encoder_real_data_trainer.train_on_batch(dataset_iterator)
            encoder_fake_data_loss = self._encoder_fake_data_trainer.train_on_batch(x=z[0], y=z[0])

            # Train Decoder
            decoder_fake_data_losses = []
            for i in range(1, decoder_steps + 1):
                decoder_fake_data_loss = self._decoder_fake_data_trainer.train_on_batch(x=z[i], y=z[i])
                decoder_fake_data_losses.append(decoder_fake_data_loss)
            decoder_fake_data_loss = float(np.mean(decoder_fake_data_losses))

            batch_logs["loss"] = encoder_real_data_loss
            batch_logs["fake_loss"] = encoder_fake_data_loss
            batch_logs["decoder_loss"] = decoder_fake_data_loss
            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(dataset, batch_size, callbacks)

    # endregion

    # region Callbacks (building)
    def callback_metrics(self):
        out_labels = self.encoder_real_data_trainer.metrics_names
        real_data_metrics = copy.copy(out_labels)
        fake_data_metrics = ["fake_{0}".format(name) for name in out_labels]
        decoder_metrics = ["decoder_{0}".format(name) for name in out_labels]
        val_metrics = ["val_{0}".format(name) for name in out_labels]
        return real_data_metrics + fake_data_metrics + decoder_metrics + val_metrics
    # endregion
