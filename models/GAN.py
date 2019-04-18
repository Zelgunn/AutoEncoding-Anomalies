import tensorflow as tf
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Reshape
from tensorflow.python.keras.callbacks import CallbackList
import numpy as np
from typing import List, Optional, Dict, Union

from models import AutoEncoderBaseModel, metrics_dict, LayerStack
from datasets import DatasetLoader


class GAN(AutoEncoderBaseModel):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator_layers: List[LayerStack] = []
        self.discriminator_regression_layer = None
        self._discriminator = None
        self._discriminator_non_trainable = None
        self._adversarial_generator = None

        self._discriminator_loss = None

        self.discriminator_depth: Optional[int] = None

        self.discriminator_input_layer: Input = None

        self._z_dataset: Optional[tf.data.Dataset] = None
        self._adversarial_generator_dataset: Optional[tf.data.Dataset] = None
        self._x_generated_datasets: Dict[int, tf.data.Dataset] = {}
        self._real_data_discriminator_datasets: Dict[int, tf.data.Dataset] = {}
        self._fake_data_discriminator_datasets: Dict[int, tf.data.Dataset] = {}

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
        autoencoded = self.decoder(self.encoder(self.encoder_input_layer))
        autoencoder = KerasModel(inputs=self.encoder_input_layer, outputs=autoencoded,
                                 name="Autoencoder")

        generator_discriminated = self.discriminator_non_trainable(self.decoder(self.decoder_input_layer))
        adversarial_generator = KerasModel(inputs=self.decoder_input_layer, outputs=generator_discriminated,
                                           name="Adversarial_Generator")

        reconstruction_metric = metrics_dict[self.config["losses"]["autoencoder"]]

        def autoencoder_loss(y_true, y_pred):
            reconstruction_loss = reconstruction_metric(y_true, y_pred) * self.config["loss_weights"]["reconstruction"]
            return reconstruction_loss

        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.compile(self.optimizer, loss=self.discriminator_loss,
                                      metrics=adversarial_generator_metrics)

        self._autoencoder = autoencoder
        self._adversarial_generator = adversarial_generator

    def compile_discriminator(self):
        self.discriminator_input_layer = Input(self.output_shape, name="discriminator_input_layer")
        layer = self.discriminator_input_layer

        for i in range(self.discriminator_depth):
            use_dropout = i < (self.discriminator_depth - 1)
            layer = self.discriminator_layers[i](layer, use_dropout)

        output_total_dim = np.prod(layer.shape[1:])
        layer = Reshape([output_total_dim], name="flatten_for_regression")(layer)
        layer = self.discriminator_regression_layer(layer)

        outputs = layer
        self._discriminator = KerasModel(inputs=self.discriminator_input_layer,
                                         outputs=outputs,
                                         name="discriminator")

        self._discriminator_non_trainable = KerasModel(inputs=self.discriminator_input_layer,
                                                       outputs=outputs,
                                                       name="discriminator_non_trainable")
        self._discriminator_non_trainable.trainable = False

        discriminator_metrics = self.config["metrics"]["discriminator"]
        self._discriminator.compile(self.optimizer, loss=self.discriminator_loss, metrics=discriminator_metrics)
    # endregion

    @property
    def discriminator(self) -> KerasModel:
        if self._discriminator is None:
            self.compile_discriminator()
        return self._discriminator

    @property
    def discriminator_non_trainable(self) -> KerasModel:
        if self._discriminator_non_trainable is None:
            self.compile_discriminator()
        return self._discriminator_non_trainable

    @property
    def adversarial_generator(self) -> KerasModel:
        if self._adversarial_generator is None:
            self.compile()
        return self._adversarial_generator

    @property
    def saved_models(self) -> Dict[str, KerasModel]:
        return {"encoder": self.encoder,
                "decoder": self.decoder,
                "discriminator": self.discriminator}

    @property
    def discriminator_loss(self):
        if self._discriminator_loss is None:
            def loss_function(y_true, y_pred):
                discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]
                return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

            self._discriminator_loss = loss_function
        return self._discriminator_loss

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
        discriminator: KerasModel = self.discriminator
        discriminator_steps = self.config["discriminator_steps"] if "discriminator_steps" in self.config else 1

        autoencoder_dataset = dataset.train_subset.tf_dataset.batch(batch_size)
        adversarial_generator_dataset = self.adversarial_generator_dataset.batch(batch_size)
        real_data_discriminator_dataset = self.get_real_data_discriminator_dataset(batch_size)
        fake_data_discriminator_dataset = self.get_fake_data_discriminator_dataset(batch_size)
        # endregion

        callbacks.on_epoch_begin(self.epochs_seen)
        # discriminator_metrics = [1.0, 0.75]
        # TODO : Turn multiple models into one
        for batch_index in range(epoch_length):
            # region Train on Batch
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            autoencoder_metrics = autoencoder.train_on_batch(autoencoder_dataset)
            generator_metrics = adversarial_generator.train_on_batch(adversarial_generator_dataset)

            discriminator_metrics = []
            for i in range(discriminator_steps):
                real_data_discriminator_metrics = discriminator.train_on_batch(real_data_discriminator_dataset)
                fake_data_discriminator_metrics = discriminator.train_on_batch(fake_data_discriminator_dataset)
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

        self.on_epoch_end(dataset, batch_size, callbacks)

    # endregion

    # region Datasets
    @property
    def z_dataset(self) -> tf.data.Dataset:
        if self._z_dataset is None:
            decoder_input_shape = self.compute_decoder_input_shape().as_list()
            decoder_input_shape = tf.constant(decoder_input_shape, dtype=tf.int32, name="decoder_input_shape")
            dataset = tf.data.Dataset.from_tensors(decoder_input_shape)
            dataset = dataset.repeat(-1)
            dataset = dataset.map(self.z_from_decoder_input_shape)
            self._z_dataset = dataset

        return self._z_dataset

    @property
    def adversarial_generator_dataset(self) -> tf.data.Dataset:
        if self._adversarial_generator_dataset is None:
            dataset = self.z_dataset.map(
                lambda z:
                (
                    (z,),
                    self.adversarial_generator_labels(shape=[1], ones=False),
                )
            )
            self._adversarial_generator_dataset = dataset

        return self._adversarial_generator_dataset

    def get_x_generated_dataset(self, batch_size: int) -> tf.data.Dataset:
        if batch_size not in self._x_generated_datasets:
            dataset = self.z_dataset.batch(batch_size)
            dataset = dataset.map(self.generate_x_from_z)
            self._x_generated_datasets[batch_size] = dataset

        return self._x_generated_datasets[batch_size]

    def get_real_data_discriminator_dataset(self, batch_size: int) -> tf.data.Dataset:
        if batch_size not in self._real_data_discriminator_datasets:
            dataset = self.get_x_generated_dataset(batch_size)
            dataset = dataset.map(
                lambda z:
                (
                    z,
                    self.adversarial_generator_labels(shape=[tf.shape(z)[0], 1], ones=False)
                )
            )
            self._real_data_discriminator_datasets[batch_size] = dataset

        return self._real_data_discriminator_datasets[batch_size]

    def get_fake_data_discriminator_dataset(self, batch_size: int) -> tf.data.Dataset:
        if batch_size not in self._fake_data_discriminator_datasets:
            dataset = self.get_x_generated_dataset(batch_size)
            dataset = dataset.map(
                lambda z:
                (
                    z,
                    self.adversarial_generator_labels(shape=[tf.shape(z)[0], 1], ones=True)
                )
            )
            self._fake_data_discriminator_datasets[batch_size] = dataset

        return self._fake_data_discriminator_datasets[batch_size]

    # endregion

    # region Datasets map functions
    @staticmethod
    def z_from_decoder_input_shape(decoder_input_shape: tf.Tensor) -> tf.Tensor:
        z = tf.random.normal(shape=decoder_input_shape, mean=0.0, stddev=1.0, name="z")
        return z

    @staticmethod
    def adversarial_generator_labels(shape: Union[List, tf.Tensor], ones: bool) -> tf.Tensor:
        mean = 0.9 if ones else 0.1
        stddev = 0.1
        min_value = 0.7 if ones else 0.0
        max_value = 1.0 if ones else 0.3

        values = tf.random.normal(shape=shape, mean=mean, stddev=stddev)
        values = tf.clip_by_value(values, min_value, max_value)
        return values

    def add_instance_noise(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        instance_noise = tf.random.normal([batch_size, *self.output_shape], stddev=0.2, mean=0.0)
        x = tf.clip_by_value(x + instance_noise, self._min_output_constant, self._max_output_constant)
        return x

    def generate_x_from_z(self, z: tf.Tensor):
        z.set_shape([None, *self.compute_decoder_input_shape()])
        x_generated = self.decoder(z)
        x_generated = self.add_instance_noise(x_generated)
        return x_generated

    # endregion

    # region Callbacks
    def callback_metrics(self):
        def model_metric_names(model_name):
            metric_names = ["loss", *self.config["metrics"][model_name]]
            return [model_name + "_" + metric_name for metric_name in metric_names]

        return model_metric_names("autoencoder") + model_metric_names("generator") + model_metric_names("discriminator")
    # endregion
