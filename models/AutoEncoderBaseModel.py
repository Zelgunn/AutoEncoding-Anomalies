import tensorflow as tf
from keras.models import Model as KerasModel
from keras.layers import Activation, LeakyReLU, Conv2D, Deconv2D, Dense, Dropout
from keras.callbacks import TensorBoard, CallbackList
from keras.utils import conv_utils
from abc import ABC, abstractmethod
import os
import json
import copy
from typing import List

from layers import ResBlock2D, ResBlock2DTranspose, SpectralNormalization
from scheme import Database, Dataset
from train_utils import get_log_dir
from callbacks import ImageCallback, AUCCallback


class AutoEncoderBaseModel(ABC):
    def __init__(self,
                 image_summaries_max_outputs: int):
        self.image_summaries_max_outputs = image_summaries_max_outputs

        self.keras_model: KerasModel = None

        self._io_delta = None
        self._error_rate = None

        self.embeddings_layer = None
        self.encoder_layers = []
        self.decoder_layers = []

        self.config: dict = None
        self.input_shape = None
        self.input_channels = None
        self.embeddings_size = None
        self.depth = 0
        self._models_per_scale = None
        self._scales_input_shapes = None
        self.default_activation = None
        self.embeddings_activation = None

        self.log_dir = None
        self.tensorboard = None

    def load_config(self, config_file):
        with open(config_file) as tmp_file:
            self.config = json.load(tmp_file)

        assert "input_shape" in self.config
        assert "embeddings_size" in self.config
        assert "encoder" in self.config
        assert "decoder" in self.config
        assert "default_activation" in self.config
        assert "embeddings_activation" in self.config

        self.input_shape = self.config["input_shape"]
        self.input_channels = self.input_shape[-1]
        self.embeddings_size = self.config["embeddings_size"]

        self.depth = len(self.config["encoder"])
        self._models_per_scale = [None] * self.depth

        self.default_activation = self.config["default_activation"]
        self.embeddings_activation = self.config["embeddings_activation"]

    # region Model building
    def build_layers(self):
        use_res_block = ("use_resblock" in self.config["use_resblock"])
        use_res_block &= self.config["use_resblock"] == "True"
        use_spectral_norm = ("use_spectral_norm" in self.config["use_spectral_norm"])
        use_spectral_norm &= self.config["use_spectral_norm"] == "True"

        for layer_info in self.config["encoder"]:
            if use_res_block:
                layer = ResBlock2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                                   use_batch_normalization=False)
            else:
                layer = Conv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                               padding=layer_info["padding"])

            if use_spectral_norm:
                layer = SpectralNormalization(layer)

            self.encoder_layers.append(layer)

        self.embeddings_layer = Dense(units=self.embeddings_size)

        i = 1
        for layer_info in self.config["decoder"]:
            if use_res_block:
                layer = ResBlock2DTranspose(layer_info["filters"], layer_info["kernel_size"],
                                            strides=layer_info["strides"])
            else:
                layer = Deconv2D(layer_info["filters"], layer_info["kernel_size"], strides=layer_info["strides"],
                                 padding=layer_info["padding"])

            if use_spectral_norm:
                layer = SpectralNormalization(layer)

            i += 1
            self.decoder_layers.append(layer)

    @abstractmethod
    def build_model(self, config_file):
        raise NotImplementedError

    @abstractmethod
    def build_model_for_scale(self, scale):
        raise NotImplementedError

    def get_model_at_scale(self, scale: int) -> KerasModel:
        if scale is None:
            model = self.keras_model
        else:
            if self._models_per_scale[scale] is None:
                self.build_model_for_scale(scale)
            model = self._models_per_scale[scale]
        return model

    def link_encoder_conv_layer(self, layer, scale, index):
        with tf.name_scope("encoder_scale_{0}".format(index)):
            # Layer
            layer_index = index + self.depth - (scale + 1)
            layer = self.encoder_layers[layer_index](layer)

            # Activation
            layer = AutoEncoderBaseModel.get_activation(self.default_activation)(layer)

            # Dropout
            if (index > self.depth - 1 - scale) and ("dropout" in self.config["encoder"][index]):
                layer = Dropout(rate=self.config["encoder"][layer_index]["dropout"],
                                name="encoder_dropout_{0}".format(index + 1))(layer)
        return layer

    def link_decoder_deconv_layer(self, layer, scale, index):
        with tf.name_scope("decoder_scale_{0}".format(index)):
            # Layer
            layer = self.decoder_layers[index](layer)

            # Activation
            layer = AutoEncoderBaseModel.get_activation(self.default_activation)(layer)

            # Dropout
            if (index != scale) and ("dropout" in self.config["decoder"][index]):
                dropout_layer_name = "decoder_dropout_{0}".format(index + 1)
                layer = Dropout(rate=self.config["decoder"][index]["dropout"], name=dropout_layer_name)(layer)
        return layer

    @staticmethod
    def get_activation(activation_config: dict):
        if activation_config["name"] == "leaky_relu":
            return LeakyReLU(alpha=activation_config["alpha"])
        else:
            return Activation(activation_config["name"])

    # endregion

    # region Training
    def train(self, database: Database, batch_size=64, epoch_length=None, epochs=1, pre_train_epochs=None, **kwargs):
        if self.log_dir is not None:
            return
        self.log_dir = self.__class__.make_log_dir(database)

        scale = kwargs.pop("scale") if "scale" in kwargs else self.depth - 1

        model = self.get_model_at_scale(scale)
        self.save_model_info(self.log_dir, model)

        samples_count = database.train_dataset.samples_count
        if epoch_length is None:
            epoch_length = samples_count // batch_size

        # region Pre-train
        common_callbacks = self.build_common_callbacks()
        common_callbacks = self.setup_callbacks(common_callbacks, model, batch_size, pre_train_epochs, epoch_length,
                                                samples_count)

        common_callbacks.on_train_begin()
        if self.can_be_pre_trained and pre_train_epochs is not None:
            self.pre_train_loop(database, common_callbacks, batch_size, epoch_length, pre_train_epochs, scale, **kwargs)
        # endregion

        # region Max scale training
        AutoEncoderBaseModel.update_callbacks_param(common_callbacks, "epochs", epochs)

        anomaly_callbacks = self.build_anomaly_callbacks(database, scale=scale)
        anomaly_callbacks = self.setup_callbacks(anomaly_callbacks, model, batch_size, pre_train_epochs, epoch_length,
                                                 samples_count)

        callbacks = CallbackList(common_callbacks.callbacks + anomaly_callbacks.callbacks)
        self.print_training_model_at_scale_header(scale, scale)

        anomaly_callbacks.on_train_begin()
        self.train_loop(database, callbacks, batch_size, epoch_length, epochs, scale, **kwargs)
        callbacks.on_train_end()
        # endregion

    @property
    def can_be_pre_trained(self):
        return False

    def pre_train_loop(self, database: Database, callbacks: CallbackList, batch_size, epoch_length, epochs, max_scale,
                       **kwargs):
        for scale in range(max_scale):
            # model = self.get_model_at_scale(scale)
            # callbacks.set_model(model)
            self.print_training_model_at_scale_header(scale, max_scale)
            self.pre_train_scale(database, callbacks, scale, batch_size, epoch_length, epochs, max_scale=max_scale,
                                 **kwargs)

    @abstractmethod
    def pre_train_scale(self, database: Database, callbacks: CallbackList, scale: int, batch_size, epoch_length, epochs,
                        **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_loop(self, database: Database, callbacks: CallbackList, batch_size, epoch_length, epochs, scale,
                   **kwargs):
        raise NotImplementedError

    def print_training_model_at_scale_header(self, scale, max_scale):
        scale_shape = self.scales_input_shapes[scale]
        tmp = len(str(scale_shape[0])) + len(str(scale_shape[1]))
        print("=============================================" + "=" * tmp)
        print("===== Training model at scale {0}x{1} (n°{2}/{3}) =====".format(scale_shape[0],
                                                                               scale_shape[1],
                                                                               scale, max_scale))
        print("=============================================" + "=" * tmp)

    # endregion

    # region Callbacks
    def build_common_callbacks(self):
        assert self.tensorboard is None
        self.tensorboard = TensorBoard(log_dir=self.log_dir, update_freq="epoch")
        return [self.tensorboard]

    def build_anomaly_callbacks(self, database, scale=None):
        scale_shape = self.scales_input_shapes[scale]
        database = database.resized_to_scale(scale_shape)

        train_image_summary_callback = self.image_summary_from_dataset(database.train_dataset, "train",
                                                                       self.tensorboard, scale=scale)
        eval_image_summary_callback = self.image_summary_from_dataset(database.test_dataset, "test",
                                                                      self.tensorboard, scale=scale)

        auc_callback = AUCCallback(self.tensorboard, self, database.test_dataset,
                                   scale=scale, plot_size=(256, 256), batch_size=128)

        return [train_image_summary_callback, eval_image_summary_callback, auc_callback]

    def setup_callbacks(self, callbacks: CallbackList or List, model: KerasModel,
                        batch_size: int, epochs: int, epoch_length: int, samples_count: int) -> CallbackList:
        callbacks = CallbackList(callbacks)
        callbacks.set_model(model)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': epoch_length,
            'samples': samples_count,
            'verbose': 1,
            'do_validation': True,
            'metrics': self.callback_metrics(model),
        })
        return callbacks

    def callback_metrics(self, model):
        metrics_names = model.metrics_names
        validation_callbacks = ["val_{0}".format(name) for name in metrics_names]
        callback_metrics = copy.copy(metrics_names) + validation_callbacks
        return callback_metrics

    def get_image_summaries(self, name):
        assert (self.keras_model is not None) and isinstance(self.keras_model, KerasModel)
        return self._get_images_summary(name, self.input, self.output, self.io_delta)

    def get_image_summaries_at_scale(self, name, scale):
        inputs = self._models_per_scale[scale].input
        outputs = self._models_per_scale[scale].output
        delta = inputs - outputs
        delta *= tf.abs(delta)

        return self._get_images_summary(name, inputs, outputs, delta)

    def image_summary_from_dataset(self, dataset: Dataset, name: str, tensorboard: TensorBoard,
                                   frequency="epoch", scale: int = None) -> ImageCallback:
        return ImageCallback.from_dataset(dataset.images,
                                          self,
                                          tensorboard,
                                          name,
                                          update_freq=frequency,
                                          scale=scale)

    def _get_images_summary(self, name, inputs, outputs, io_delta):
        summaries = [tf.summary.image(name + "_inputs", inputs,
                                      max_outputs=self.image_summaries_max_outputs),
                     tf.summary.image(name + "_outputs", outputs,
                                      max_outputs=self.image_summaries_max_outputs),
                     tf.summary.image(name + "_delta", io_delta,
                                      max_outputs=self.image_summaries_max_outputs)
                     ]
        return tf.summary.merge(summaries)

    def frame_level_average_error(self, scale=None, normalize_error=True):
        model = self.get_model_at_scale(scale)

        squared_delta = tf.square(model.input - model.output)
        average_error = tf.reduce_mean(squared_delta, axis=[1, 2, 3])
        average_error = tf.sqrt(average_error)

        if normalize_error:
            min_error = tf.reduce_min(average_error)
            max_error = tf.reduce_max(average_error)
            error_range = max_error - min_error + tf.constant(1e-7)

            average_error = (average_error - min_error) / error_range

        return average_error

    @staticmethod
    def update_callbacks_param(callbacks: CallbackList, param_name: str, param_value):
        for callback in callbacks:
            callback.params[param_name] = param_value
            if hasattr(callback, param_name):
                setattr(callback, param_name, param_value)

    # endregion

    @property
    def scales_input_shapes(self):
        if self._scales_input_shapes is None:
            input_shape = self.input_shape
            self._scales_input_shapes = []

            for layer_info in self.config["encoder"]:
                self._scales_input_shapes.append(input_shape)

                space = input_shape[:-1]
                kernel_size = conv_utils.normalize_tuple(layer_info["kernel_size"], 2, "kernel_size")
                strides = conv_utils.normalize_tuple(layer_info["strides"], 2, "strides")

                new_space = []
                for i in range(len(space)):
                    dim = conv_utils.conv_output_length(space[i],
                                                        kernel_size[i],
                                                        padding=layer_info["padding"],
                                                        stride=strides[i])
                    new_space.append(dim)
                input_shape = [*new_space, layer_info["filters"]]

            self._scales_input_shapes.reverse()
        return self._scales_input_shapes

    # region Log dir

    @classmethod
    def make_log_dir(cls, database: Database):
        project_log_dir = "../logs/AnomalyBasicModelsBenchmark"
        base_dir = os.path.join(project_log_dir, cls.__name__, database.__class__.__name__)
        log_dir = get_log_dir(base_dir)
        return log_dir

    def save_model_info(self, log_dir, model: KerasModel = None):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if model is None:
            model = self.keras_model

        keras_config_filename = os.path.join(log_dir, "{0}_keras_config.json".format(model.name))
        with open(keras_config_filename, "w") as file:
            file.write(model.to_json())

        summary_filename = os.path.join(log_dir, "{0}_summary.txt".format(model.name))
        with open(summary_filename, "w") as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        config_filename = os.path.join(log_dir, "{0}_config.json".format(model.name))
        with open(config_filename, "w") as file:
            json.dump(self.config, file)

    # endregion

    # region Legacy properties
    @property
    def input(self):
        return self.keras_model.input

    @property
    def output(self):
        return self.keras_model.output

    @property
    def io_delta(self):
        if self._io_delta is None:
            self._io_delta = self.input - self.output
        return self._io_delta

    @property
    def error_rate(self):
        if self._error_rate is None:
            self._error_rate = tf.reduce_mean(tf.abs(self.io_delta))
        return self._error_rate

    # endregion