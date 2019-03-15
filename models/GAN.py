from keras.layers import Input, Dense, Reshape
from keras.callbacks import CallbackList
import tensorflow as tf
import numpy as np
from typing import List

from models import AutoEncoderBaseModel, KerasModel, metrics_dict, LayerStack
from datasets import Database


class GAN(AutoEncoderBaseModel):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator_layers: List[LayerStack] = []
        self.discriminator_regression_layer = None
        self._discriminator = None
        self._adversarial_generator = None

    def build_layers(self):
        super(GAN, self).build_layers()

        for stack_info in self.config["discriminator"]:
            stack = self.build_conv_stack(stack_info)
            self.discriminator_layers.append(stack)

        self.discriminator_regression_layer = Dense(units=1, activation="sigmoid")

    # region Compile
    def compile(self):
        encoder_input = Input(self.input_shape)
        autoencoded = self.decoder(self.encoder(encoder_input))
        autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoded,
                                 name="Autoencoder")

        decoder_input = Input(self.compute_decoder_input_shape())
        generator_discriminated = self.discriminator(self.decoder(decoder_input))
        adversarial_generator = KerasModel(inputs=decoder_input, outputs=generator_discriminated,
                                           name="Adversarial_Generator")

        discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]

        def discriminator_loss(y_true, y_pred):
            return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

        discriminator_metrics = self.config["metrics"]["discriminator"]
        self.discriminator.compile(self.optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

        reconstruction_metric = metrics_dict[self.config["losses"]["autoencoder"]]

        def autoencoder_loss(y_true, y_pred):
            reconstruction_loss = reconstruction_metric(y_true, y_pred) * self.config["loss_weights"]["reconstruction"]
            return reconstruction_loss

        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        self.discriminator.trainable = False
        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.compile(self.optimizer, loss=discriminator_loss,
                                      metrics=adversarial_generator_metrics)

        self._autoencoder = autoencoder
        self._adversarial_generator = adversarial_generator

    def compile_discriminator(self):
        discriminator_name = "Discriminator"
        input_layer = Input(self.input_shape)
        layer = input_layer

        for i in range(self.depth):
            use_dropout = i < (self.depth - 1)
            layer = self.discriminator_layers[i](input_layer, use_dropout)

        layer = Reshape([-1])(layer)
        layer = self.discriminator_regression_layer(layer)

        outputs = layer
        self._discriminator = KerasModel(inputs=input_layer, outputs=outputs, name=discriminator_name)

    # endregion

    @property
    def discriminator(self):
        if self._discriminator is None:
            self.compile_discriminator()
        return self._discriminator

    @property
    def adversarial_generator(self):
        if self._adversarial_generator is None:
            self.compile()
        return self._adversarial_generator

    @property
    def saved_models(self):
        return {"encoder": self.encoder,
                "decoder": self.decoder,
                "discriminator": self.discriminator}

    # region Training
    def train_epoch(self, database: Database, callbacks: CallbackList = None):
        # region Variables initialization
        epoch_length = len(database.train_subset)
        autoencoder: KerasModel = self.autoencoder
        decoder: KerasModel = self.decoder
        adversarial_generator: KerasModel = self.adversarial_generator
        discriminator: KerasModel = self.discriminator
        discriminator_steps = self.config["discriminator_steps"] if "discriminator_steps" in self.config else 1
        # endregion

        callbacks.on_epoch_begin(self.epochs_seen)
        # discriminator_metrics = [1.0, 0.75]
        for batch_index in range(epoch_length):
            # region Generate batch data (common)
            noisy_x, x = database.train_subset[0]
            batch_size = x.shape[0]
            x_real = [x]
            z = np.random.normal(size=[discriminator_steps, batch_size, *self.compute_decoder_input_shape()])
            zeros = np.random.normal(size=[discriminator_steps, batch_size], loc=0.1, scale=0.1)
            zeros = np.clip(zeros, 0.0, 0.3)
            ones = np.random.normal(size=[discriminator_steps, batch_size], loc=0.9, scale=0.1)
            ones = np.clip(ones, 0.7, 1.0)
            # endregion

            # region Generate batch data (discriminator)
            x_generated = []
            for i in range(discriminator_steps):
                if i > 0:
                    x_real += [database.train_subset.sample(sequence_length=database.output_sequence_length)]
                x_generated += [decoder.predict(x=z[i])]

            x_real = np.array(x_real)
            x_generated = np.array(x_generated)

            instance_noise = np.random.normal(size=[2, *x_real.shape], scale=0.2)

            x_real = np.clip(x_real + instance_noise[0], -1.0, 1.0)
            x_generated = np.clip(x_generated + instance_noise[1], -1.0, 1.0)
            # endregion

            # region Train on Batch
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            autoencoder_metrics = autoencoder.train_on_batch(x=noisy_x, y=x_real[0])
            generator_metrics = adversarial_generator.train_on_batch(x=z[0], y=zeros[0])

            discriminator_metrics = []
            for i in range(discriminator_steps):
                real_data_discriminator_metrics = discriminator.train_on_batch(x=x_real[i], y=zeros[i])
                fake_data_discriminator_metrics = discriminator.train_on_batch(x=x_generated[i], y=ones[i])
                discriminator_metrics += [real_data_discriminator_metrics, fake_data_discriminator_metrics]
            discriminator_metrics = np.mean(discriminator_metrics, axis=0)

            # region Batch logs
            def add_metrics_to_batch_logs(model_name, losses):
                metric_names = ["loss", *self.config["metrics"][model_name]]
                if hasattr(losses, "__len__"):
                    for j in range(len(losses)):
                        batch_logs[model_name + '_' + metric_names[j]] = losses[j]
                else:
                    batch_logs[model_name + '_' + metric_names[0]] = losses

            add_metrics_to_batch_logs("autoencoder", autoencoder_metrics)
            add_metrics_to_batch_logs("generator", generator_metrics)
            add_metrics_to_batch_logs("discriminator", discriminator_metrics)
            # endregion

            callbacks.on_batch_end(batch_index, batch_logs)
            # endregion

        self.on_epoch_end(database, callbacks)

    # endregion

    # region Callbacks
    def callback_metrics(self):
        def model_metric_names(model_name):
            metric_names = ["loss", *self.config["metrics"][model_name]]
            return [model_name + "_" + metric_name for metric_name in metric_names]

        return model_metric_names("autoencoder") + model_metric_names("generator") + model_metric_names("discriminator")
    # endregion
