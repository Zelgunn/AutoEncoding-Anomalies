import tensorflow as tf
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Reshape
from tensorflow.python.keras.callbacks import CallbackList
import numpy as np
from typing import List, Optional, Dict

from models import AutoEncoderBaseModel, metrics_dict, LayerStack
from datasets import DatasetLoader
from modalities import Pattern


class GAN(AutoEncoderBaseModel):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator_layers: List[LayerStack] = []
        self.discriminator_regression_layer = None
        self._discriminator: Optional[KerasModel] = None
        self._adversarial_generator: Optional[KerasModel] = None
        self._adversarial_discriminator: Optional[KerasModel] = None

        self.discriminator_depth: Optional[int] = None

        self.discriminator_input_layer: Input = None

        self._z_dataset: Optional[tf.data.Dataset] = None
        self._real_data_discriminator_dataset: Optional[tf.data.Dataset] = None

        self._discriminator_loss_real_data = None
        self._discriminator_loss_fake_data = None

        self._generator_z_iterator = None
        self._discriminator_z_iterator = None
        self._real_x_iterator = None

    def load_config(self, config_file: str, alt_config_file: str):
        super(GAN, self).load_config(config_file, alt_config_file)

        self.discriminator_depth = len(self.config["discriminator"])

    def build_layers(self):
        super(GAN, self).build_layers()

        for stack_info in self.config["discriminator"]:
            stack = self.build_conv_stack(stack_info)
            self.discriminator_layers.append(stack)

        self.discriminator_regression_layer = Dense(units=1, activation="sigmoid")

    # region Compile
    def compile(self):
        # region KerasModel (init)
        autoencoded = self.decoder(self.encoder(self.encoder_input_layer))
        autoencoder = KerasModel(inputs=self.encoder_input_layer, outputs=autoencoded,
                                 name="Autoencoder")

        # region Adversarial generator (train generator with discriminator)
        decoder_trainable = KerasModel(inputs=self.decoder.input,
                                       outputs=self.decoder.output,
                                       name="decoder_trainable")
        decoder_trainable.trainable = True

        discriminator_non_trainable = KerasModel(inputs=self.discriminator.input,
                                                 outputs=self.discriminator.output,
                                                 name="discriminator_non_trainable")
        discriminator_non_trainable.trainable = False

        adversarial_generator_output = discriminator_non_trainable(decoder_trainable(self.decoder_input_layer))
        adversarial_generator = KerasModel(inputs=self.decoder_input_layer,
                                           outputs=adversarial_generator_output,
                                           name="Adversarial_Generator")
        # endregion

        # region Adversarial discriminator (train discriminator with fake data)
        decoder_non_trainable = KerasModel(inputs=self.decoder.input,
                                           outputs=self.decoder.output,
                                           name="decoder_non_trainable")
        decoder_non_trainable.trainable = False

        discriminator_trainable = KerasModel(inputs=self.discriminator.input,
                                             outputs=self.discriminator.output,
                                             name="discriminator_trainable")
        discriminator_trainable.trainable = True

        adversarial_discriminator_output = discriminator_trainable(decoder_non_trainable(self.decoder_input_layer))
        adversarial_discriminator = KerasModel(inputs=self.decoder_input_layer,
                                               outputs=adversarial_discriminator_output,
                                               name="Adversarial_Discriminator")
        # endregion
        # endregion

        # region KerasModel (compile)
        reconstruction_metric = self.get_reconstruction_loss(self.config["losses"]["autoencoder"])

        def autoencoder_loss(y_true, y_pred):
            reconstruction_loss = reconstruction_metric(y_true, y_pred) * self.config["loss_weights"]["reconstruction"]
            return reconstruction_loss

        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.add_loss(self.discriminator_loss_real_data(adversarial_generator_output))
        adversarial_generator.compile(self.optimizer,
                                      metrics=adversarial_generator_metrics)

        adversarial_discriminator_metrics = self.config["metrics"]["discriminator"]
        adversarial_discriminator.add_loss(self.discriminator_loss_fake_data(adversarial_discriminator_output))
        adversarial_discriminator.compile(self.optimizer,
                                          metrics=adversarial_discriminator_metrics)

        # endregion

        self._autoencoder = autoencoder
        self._adversarial_generator = adversarial_generator
        self._adversarial_discriminator = adversarial_discriminator

    def compile_discriminator(self):
        self.discriminator_input_layer = Input(self.output_shape, name="discriminator_input_layer")
        layer = self.discriminator_input_layer

        for i in range(self.discriminator_depth):
            use_dropout = i < (self.discriminator_depth - 1)
            layer = self.discriminator_layers[i](layer, use_dropout)

        output_total_dim = np.prod(layer.shape[1:].as_list())
        layer = Reshape([output_total_dim], name="flatten_for_regression")(layer)
        layer = self.discriminator_regression_layer(layer)

        outputs = layer
        self._discriminator = KerasModel(inputs=self.discriminator_input_layer,
                                         outputs=outputs,
                                         name="discriminator")

        discriminator_metrics = self.config["metrics"]["discriminator"]
        self._discriminator.add_loss(self.discriminator_loss_real_data(outputs))
        self._discriminator.compile(self.optimizer, metrics=discriminator_metrics)

    # region Discriminator loss(es)
    @property
    def discriminator_loss(self):
        def loss_function(y_pred, fake_data: bool):
            mean = 0.9 if fake_data else 0.1
            stddev = 0.1
            min_value = 0.7 if fake_data else 0.0
            max_value = 1.0 if fake_data else 0.3

            y_true = tf.random.normal(shape=tf.shape(y_pred), mean=mean, stddev=stddev)
            y_true = tf.clip_by_value(y_true, min_value, max_value)

            discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]
            return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

        return loss_function

    @property
    def discriminator_loss_real_data(self):
        if self._discriminator_loss_real_data is None:
            def loss_function(y_pred):
                return self.discriminator_loss(y_pred, False)

            self._discriminator_loss_real_data = loss_function
        return self._discriminator_loss_real_data

    @property
    def discriminator_loss_fake_data(self):
        if self._discriminator_loss_fake_data is None:
            def loss_function(y_pred):
                return self.discriminator_loss(y_pred, True)

            self._discriminator_loss_fake_data = loss_function
        return self._discriminator_loss_fake_data

    # endregion
    # endregion

    # region Models
    @property
    def discriminator(self) -> KerasModel:
        if self._discriminator is None:
            self.compile_discriminator()
        return self._discriminator

    @property
    def adversarial_generator(self) -> KerasModel:
        if self._adversarial_generator is None:
            self.compile()
        return self._adversarial_generator

    @property
    def adversarial_discriminator(self) -> KerasModel:
        if self._adversarial_discriminator is None:
            self.compile()
        return self._adversarial_discriminator

    @property
    def saved_models(self) -> Dict[str, KerasModel]:
        return {"encoder": self.encoder,
                "decoder": self.decoder,
                "discriminator": self.discriminator}

    @property
    def models_with_saved_info(self) -> List[KerasModel]:
        gan_models = [self.discriminator, self.adversarial_generator, self.adversarial_discriminator]
        return super(GAN, self).models_with_saved_info + gan_models

    # endregion

    # region Training
    def train_epoch(self,
                    dataset: DatasetLoader,
                    callbacks: CallbackList,
                    batch_size: int,
                    epoch_length: int
                    ):
        # region Variables initialization
        autoencoder: KerasModel = self.autoencoder
        adversarial_generator: KerasModel = self.adversarial_generator
        adversarial_discriminator: KerasModel = self.adversarial_discriminator
        discriminator: KerasModel = self.discriminator
        discriminator_steps = self.config["discriminator_steps"] if "discriminator_steps" in self.config else 1

        generator_z_iterator = self._generator_z_iterator
        real_x_iterator = self._real_x_iterator
        discriminator_z_iterator = self._discriminator_z_iterator
        # endregion

        callbacks.on_epoch_begin(self.epochs_seen)
        for batch_index in range(epoch_length):
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)
            autoencoder_metrics = autoencoder.train_on_batch(self._train_dataset_iterator)
            generator_metrics = adversarial_generator.train_on_batch(generator_z_iterator)

            discriminator_metrics = []
            for i in range(discriminator_steps):
                real_data_discriminator_metrics = discriminator.train_on_batch(real_x_iterator)
                fake_data_discriminator_metrics = adversarial_discriminator.train_on_batch(discriminator_z_iterator)
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

        self.on_epoch_end(callbacks)

    # region Make dataset iterators
    def make_dataset_iterators(self, dataset: DatasetLoader, batch_size: int):
        if self._train_dataset_iterator is not None:
            return
        
        super(GAN, self).make_dataset_iterators(dataset, batch_size)

        self._generator_z_iterator = self.batch_and_prefetch(self.z_dataset, batch_size, prefetch=False)
        self._discriminator_z_iterator = self.batch_and_prefetch(self.z_dataset, batch_size, prefetch=False)

        pattern = Pattern()
        if len(pattern) == 0:
            raise NotImplementedError

        real_x_dataset = self.get_real_data_discriminator_dataset(dataset.train_subset.make_tf_dataset(pattern))
        self._real_x_iterator = self.batch_and_prefetch(real_x_dataset, batch_size, prefetch=False)
    # endregion

    # endregion

    # region Datasets
    @property
    def z_dataset(self) -> tf.data.Dataset:
        if self._z_dataset is None:
            decoder_input_shape = self.compute_decoder_input_shape()
            decoder_input_shape = tf.constant(decoder_input_shape, dtype=tf.int32, name="decoder_input_shape")
            dataset = tf.data.Dataset.from_tensors(decoder_input_shape)
            dataset = dataset.repeat(-1)
            dataset = dataset.map(self.z_from_decoder_input_shape)
            self._z_dataset = dataset

        return self._z_dataset

    def get_real_data_discriminator_dataset(self, x_dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self._real_data_discriminator_dataset is None:
            dataset = x_dataset.map(lambda x, y: y[0])
            self._real_data_discriminator_dataset = dataset
        return self._real_data_discriminator_dataset

    @staticmethod
    def z_from_decoder_input_shape(decoder_input_shape: tf.Tensor) -> tf.Tensor:
        z = tf.random.normal(shape=decoder_input_shape, mean=0.0, stddev=1.0, name="z")
        return z

    def add_instance_noise(self, x: tf.Tensor) -> tf.Tensor:
        instance_noise = tf.random.normal(self.output_shape, stddev=0.2, mean=0.0)
        x = tf.clip_by_value(x + instance_noise, self._min_output_constant, self._max_output_constant)
        return x

    # endregion

    # region Callbacks
    def callback_metrics(self):
        def model_metric_names(model_name):
            metric_names = ["loss", *self.config["metrics"][model_name]]
            return [model_name + "_" + metric_name for metric_name in metric_names]

        return model_metric_names("autoencoder") + model_metric_names("generator") + model_metric_names("discriminator")
    # endregion
