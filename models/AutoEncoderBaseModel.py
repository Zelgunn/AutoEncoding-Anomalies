from keras.models import Model as KerasModel
from keras.layers import Activation, LeakyReLU, Dense, Dropout, Layer, Input, Reshape, Concatenate, BatchNormalization
from keras.layers import Conv3D, Deconv3D, MaxPooling3D, AveragePooling3D, UpSampling3D
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, CallbackList, Callback, ProgbarLogger, BaseLogger, LearningRateScheduler
from keras.backend import binary_crossentropy, get_session
from keras.utils.generic_utils import to_list
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
import json
import copy
from tqdm import tqdm
from typing import List, Any

from layers import ResBlock3D, ResBlock3DTranspose, DenseBlock3D, SpectralNormalization
from datasets import Database, Dataset
from utils.train_utils import get_log_dir
from utils.summary_utils import image_summary
from callbacks import ImageCallback, AUCCallback, CallbackModel
from data_preprocessors import DropoutNoiser, BrightnessShifter, RandomCropper


# region Containers
class LayerBlock(object):
    def __init__(self,
                 conv_layer: Layer,
                 pool_layer: Layer,
                 upsampling_layer: Layer,
                 activation_layer: Layer,
                 dropout_layer: Dropout,
                 is_transpose: bool,
                 add_batch_normalization=False):
        self.conv = conv_layer
        self.pooling = pool_layer
        self.upsampling_layer = upsampling_layer
        self.activation = activation_layer
        self.dropout = dropout_layer

        self.is_transpose = is_transpose
        self.add_batch_normalization = add_batch_normalization

    def __call__(self, input_layer, use_dropout):
        with tf.name_scope("block"):
            use_dropout &= self.dropout is not None
            layer = input_layer

            if use_dropout and self.is_transpose:
                layer = self.dropout(layer)

            if self.upsampling_layer is not None:
                assert self.is_transpose
                layer = self.upsampling_layer(layer)

            layer = self.conv(layer)

            if self.add_batch_normalization:
                layer = BatchNormalization()(layer)

            if self.activation is not None:
                layer = self.activation(layer)

            if self.pooling is not None:
                assert not self.is_transpose
                layer = self.pooling(layer)

            if use_dropout and not self.is_transpose:
                layer = self.dropout(layer)

        return layer

    def compute_output_shape(self, input_shape):
        if self.upsampling_layer is not None:
            input_shape = self.upsampling_layer.compute_output_shape(input_shape)

        input_shape = self.conv.compute_output_shape(input_shape)

        if self.pooling is not None:
            input_shape = self.pooling.compute_output_shape(input_shape)

        return input_shape


class LayerStack(object):
    def __init__(self):
        self.layers: List[LayerBlock] = []

    def add_layer(self, layer_block: LayerBlock):
        self.layers.append(layer_block)

    @property
    def depth(self):
        return len(self.layers)

    def __call__(self, input_layer: tf.Tensor, use_dropout: bool):
        output_layer = input_layer

        for layer in self.layers:
            output_layer = layer(output_layer, use_dropout)

        return output_layer

    def compute_output_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer.compute_output_shape(input_shape)
        return input_shape


# endregion

# region Output activations

output_activation_ranges = {"sigmoid": [0.0, 1.0],
                            "tanh": [-1.0, 1.0]}


# endregion

# region Reconstruction metrics
def absolute_error(y_true, y_pred, axis=None):
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(error, axis)


def squared_error(y_true, y_pred, axis=None):
    error = tf.square(y_true - y_pred)
    return tf.reduce_mean(error, axis)


def cosine_distance(y_true, y_pred, axis=None):
    with tf.name_scope("Cosine_distance"):
        y_true = tf.nn.l2_normalize(y_true, axis=axis)
        y_pred = tf.nn.l2_normalize(y_pred, axis=axis)
        distance = 2.0 - y_true * y_pred
        return tf.reduce_mean(distance, axis)


def mean_binary_crossentropy(y_true, y_pred, axis=None):
    return tf.reduce_mean(binary_crossentropy(y_true, y_pred), axis=axis)


metrics_dict = {"L1": absolute_error,
                "L2": squared_error,
                "cos": cosine_distance,
                "bce": mean_binary_crossentropy}

# endregion

# region Dynamic choice between Conv/ResBlock/DenseBlock/MaxPooling/...
types_with_transpose = ["conv_block", "residual_block"]

conv_type = {"conv_block": {False: Conv3D, True: Deconv3D},
             "residual_block": {False: ResBlock3D, True: ResBlock3DTranspose},
             "dense_block": {False: DenseBlock3D}}

pool_type = {"max": MaxPooling3D,
             "average": AveragePooling3D}
# endregion

uint8_max_constant = tf.constant(255.0, dtype=tf.float32)


class AutoEncoderBaseModel(ABC):
    # region Initialization
    def __init__(self):
        self.image_summaries_max_outputs = 3

        self.layers_built = False
        self.embeddings_layer: Layer = None
        self.encoder_layers: List[LayerStack] = []
        self.reconstructor_layers: List[LayerStack] = []
        self.predictor_layers: List[LayerStack] = []

        self._encoder: KerasModel = None
        self._decoder: KerasModel = None
        self._autoencoder: KerasModel = None

        self.decoder_input_layer = None
        self.reconstructor_output = None
        self.predictor_output = None

        self.config: dict = None
        self.depth: int = None

        self.input_shape = None
        self.output_shape = None
        self.channels_count = None

        self.embeddings_size = None
        self.use_dense_embeddings = None

        self._true_outputs_placeholder: tf.Tensor = None
        self._auc_predictions_placeholder: tf.Tensor = None
        self._auc_labels_placeholder: tf.Tensor = None
        self._auc_ops: List[tf.Tensor, tf.Tensor] = None

        self.default_activation = None
        self.embeddings_activation = None
        self.output_activation = None

        self.weights_initializer = None

        self.block_type_name = None
        self.pooling = None
        self.upsampling = None
        self.use_batch_normalization = None

        self.temporal_loss_weights = None
        self.optimizer = None

        self.log_dir = None
        self.tensorboard = None
        self.epochs_seen = 0
        self.pixel_level_labels_size = None
        self.output_range = None
        self._min_output_constant: tf.Tensor = None
        self._inv_output_range_constant: tf.Tensor = None
        self.use_spectral_norm = False
        self.weight_decay_regularizer = None

        self.train_data_preprocessors = None
        self.test_data_preprocessors = None

    def load_config(self, config_file: str, alt_config_file: str or None):
        # region Open/Load json file(s)
        with open(config_file) as tmp_file:
            self.config = json.load(tmp_file)

        if alt_config_file is not None:
            with open(alt_config_file) as tmp_file:
                alt_config = json.load(tmp_file)

            for key in alt_config:
                self.config[key] = alt_config[key]
        # endregion

        # region Embeddings
        self.embeddings_size = self.config["embeddings_size"]
        self.use_dense_embeddings = self.config["use_dense_embeddings"] == "True"
        # endregion

        self.depth = len(self.config["encoder"])
        self._true_outputs_placeholder = None

        # region I/O shapes
        self.input_shape = self.config["input_shape"]
        assert 3 <= len(self.input_shape) <= 4
        self.channels_count = self.input_shape[-1]
        self.output_shape = copy.copy(self.input_shape)
        self.output_shape[0] *= 2
        # endregion

        # region Activations
        self.default_activation = self.config["default_activation"]
        self.embeddings_activation = self.config["embeddings_activation"]
        self.output_activation = self.config["output_activation"]
        # endregion

        # region Weights initialization
        weights_initializer_config = self.config["weights_initializer"]
        weights_initializer_seed = weights_initializer_config["seed"] if "seed" in weights_initializer_config else None
        self.weights_initializer = VarianceScaling(scale=weights_initializer_config["scale"],
                                                   mode=weights_initializer_config["mode"],
                                                   distribution=weights_initializer_config["distribution"],
                                                   seed=weights_initializer_seed)

        # endregion

        self.block_type_name = self.config["block_type"]
        self.pooling = self.config["pooling"]
        self.upsampling = self.config["upsampling"]
        self.use_batch_normalization = (self.config["use_batch_normalization"] == "True")

        # region Regularizers
        self.weight_decay_regularizer = l2(self.config["weight_decay"]) if "weight_decay" in self.config else None
        self.use_spectral_norm = ("use_spectral_norm" in self.config["use_spectral_norm"])
        self.use_spectral_norm &= self.config["use_spectral_norm"] == "True"
        # endregion

        # region Optimizer
        self.build_optimizer()
        # endregion

        # region Callbacks
        self.pixel_level_labels_size = tuple(self.config["pixel_level_labels_size"])
        self.output_range = output_activation_ranges[self.output_activation["name"]]
        self._min_output_constant = tf.constant(self.output_range[0], name="min_output")
        inv_range = 1.0 / (self.output_range[1] - self.output_range[0])
        self._inv_output_range_constant = tf.constant(inv_range, name="inv_output_range")
        # endregion

        self.train_data_preprocessors = self.get_data_preprocessors(True)
        self.test_data_preprocessors = self.get_data_preprocessors(False)

    # region Data Preprocessors
    def get_data_preprocessors(self, train: bool):
        data_preprocessors = []

        section_name = "train" if train else "test"
        section = self.config["data_generators"][section_name]

        if "cropping" in section:
            cropping_section = section["cropping"]
            width_range = cropping_section["width_range"]
            height_range = cropping_section["height_range"] if "height_range" in cropping_section else None
            keep_ratio = (cropping_section["keep_ratio"] == "True") if "keep_ratio" in cropping_section else True
            random_cropper = RandomCropper(width_range, height_range, keep_ratio)
            data_preprocessors.append(random_cropper)

        if "brightness" in section:
            brightness_section = section["brightness"]
            gain = brightness_section["gain"] if "gain" in brightness_section else None
            bias = brightness_section["bias"] if "bias" in brightness_section else None
            norm_method = brightness_section["normalization_method"] if "normalization_method" in brightness_section \
                else "clip"
            brightness_shifter = BrightnessShifter(gain=gain, bias=bias, values_range=self.output_range,
                                                   normalization_method=norm_method)
            data_preprocessors.append(brightness_shifter)

        if "dropout_rate" in section:
            dropout_noise_rate = section["dropout_rate"]
            dropout_noiser = DropoutNoiser(inputs_dropout_rate=dropout_noise_rate)
            data_preprocessors.append(dropout_noiser)

        return data_preprocessors

    # endregion

    # endregion

    # region Build
    def build_layers(self):
        for stack_info in self.config["encoder"]:
            stack = self.build_conv_stack(stack_info)
            self.encoder_layers.append(stack)

        if self.use_dense_embeddings:
            self.embeddings_layer = Dense(units=self.embeddings_size,
                                          kernel_regularizer=self.weight_decay_regularizer,
                                          bias_regularizer=self.weight_decay_regularizer)
        else:
            conv = conv_type["conv_block"][False]
            self.embeddings_layer = conv(filters=self.embeddings_size, kernel_size=3, padding="same",
                                         kernel_initializer=self.weights_initializer,
                                         kernel_regularizer=self.weight_decay_regularizer,
                                         bias_regularizer=self.weight_decay_regularizer)

        for stack_info in self.config["decoder"]:
            stack = self.build_deconv_stack(stack_info)
            self.reconstructor_layers.append(stack)
            stack = self.build_deconv_stack(stack_info)
            self.predictor_layers.append(stack)
        self.layers_built = True

    def build_conv_stack(self, stack_info: dict, transpose=False) -> LayerStack:
        if self.block_type_name == "dense_block":
            return self.build_dense_block(stack_info, transpose)
        else:
            return self.build_layer_stack(stack_info, transpose)

    def build_deconv_stack(self, stack_info: dict) -> LayerStack:
        return self.build_conv_stack(stack_info, transpose=True)

    def build_dense_block(self, stack_info: dict, transpose=False) -> LayerStack:
        depth: int = stack_info["depth"]
        stack = LayerStack()

        conv_block_type = self.get_block_type(transpose)
        conv_layer = conv_block_type(kernel_size=stack_info["kernel_size"],
                                     growth_rate=12,
                                     depth=depth,
                                     output_filters=stack_info["filters"],
                                     use_batch_normalization=self.use_batch_normalization,
                                     use_bias=True,
                                     kernel_initializer=self.weights_initializer,
                                     kernel_regularizer=self.weight_decay_regularizer,
                                     bias_regularizer=self.weight_decay_regularizer
                                     )

        if self.use_spectral_norm:
            conv_layer = SpectralNormalization(conv_layer)

        size = stack_info["strides"]
        if transpose:
            pool_layer = None
            upsampling_layer = UpSampling3D(size=size)
        else:
            pool_layer = pool_type[self.pooling](pool_size=size)
            upsampling_layer = None

        activation = AutoEncoderBaseModel.get_activation(self.default_activation)
        dropout = Dropout(rate=stack_info["dropout"]) if "dropout" in stack_info else None

        layer_block = LayerBlock(conv_layer, pool_layer, upsampling_layer, activation, dropout, transpose)
        stack.add_layer(layer_block)

        return stack

    def build_layer_stack(self, stack_info: dict, transpose=False) -> LayerStack:
        depth: int = stack_info["depth"]
        stack = LayerStack()

        for i in range(depth):

            # region Pooling/Upsampling layer
            if i < (depth - 1):
                conv_layer_strides = 1
                pool_layer = None
                upsampling_layer = None
            elif self.pooling == "strides":
                conv_layer_strides = stack_info["strides"]
                pool_layer = None
                upsampling_layer = None
            else:
                conv_layer_strides = 1
                size = stack_info["strides"]
                if transpose:
                    pool_layer = None
                    upsampling_layer = UpSampling3D(size=size)
                else:
                    pool_layer = pool_type[self.pooling](pool_size=size)
                    upsampling_layer = None
            # endregion

            # region Convolutional layer
            conv_layer_kwargs = {"kernel_size": stack_info["kernel_size"],
                                 "filters": stack_info["filters"],
                                 "strides": conv_layer_strides,
                                 "kernel_initializer": self.weights_initializer,
                                 "kernel_regularizer": self.weight_decay_regularizer,
                                 "bias_regularizer": self.weight_decay_regularizer}

            if self.block_type_name == "residual_block":
                conv_layer_kwargs["use_batch_normalization"] = self.use_batch_normalization
                add_batch_normalization = False
            else:
                conv_layer_kwargs["padding"] = stack_info["padding"] if "padding" in stack_info else "same"
                add_batch_normalization = self.use_batch_normalization

            conv_block_type = self.get_block_type(transpose)
            conv_layer = conv_block_type(**conv_layer_kwargs)
            # endregion

            # region Spectral normalization
            if self.use_spectral_norm:
                conv_layer = SpectralNormalization(conv_layer)
            # endregion

            activation = AutoEncoderBaseModel.get_activation(self.default_activation)
            dropout = Dropout(rate=stack_info["dropout"]) if "dropout" in stack_info else None

            layer_block = LayerBlock(conv_layer, pool_layer, upsampling_layer, activation, dropout, transpose,
                                     add_batch_normalization=add_batch_normalization)
            stack.add_layer(layer_block)

        return stack

    def get_block_type(self, transpose: bool):
        if self.block_type_name not in types_with_transpose:
            transpose = False

        return conv_type[self.block_type_name][transpose]

    def compute_embeddings_input_shape(self):
        embeddings_shape = (None, *self.input_shape)

        for stack in self.encoder_layers:
            embeddings_shape = stack.compute_output_shape(embeddings_shape)

        return embeddings_shape[1:]

    def compute_embeddings_output_shape(self):
        embeddings_shape = self.compute_embeddings_input_shape()
        if self.use_dense_embeddings:
            embeddings_shape = embeddings_shape[:-1]
            embeddings_shape_prod = np.prod(embeddings_shape)
            assert (self.embeddings_size % embeddings_shape_prod) == 0
            embeddings_filters = self.embeddings_size // embeddings_shape_prod
            embeddings_shape = (*embeddings_shape, embeddings_filters)
        else:
            embeddings_shape = (None, *embeddings_shape)
            embeddings_shape = self.embeddings_layer.compute_output_shape(embeddings_shape)
            embeddings_shape = embeddings_shape[1:]
        return embeddings_shape

    def compute_decoder_input_shape(self):
        return self.compute_embeddings_output_shape()

    def build_optimizer(self):
        optimizer_name = self.config["optimizer"]["name"].lower()
        if optimizer_name == "adam":
            self.optimizer = Adam(lr=self.config["optimizer"]["lr"],
                                  beta_1=self.config["optimizer"]["beta_1"],
                                  beta_2=self.config["optimizer"]["beta_2"],
                                  decay=self.config["optimizer"]["decay"])
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSprop(lr=self.config["optimizer"]["lr"],
                                     rho=self.config["optimizer"]["rho"],
                                     decay=self.config["optimizer"]["decay"])
        else:
            raise ValueError

    # endregion

    # region Compile

    @abstractmethod
    def compile(self):
        raise NotImplementedError

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

        output_layer = layer

        self._encoder = KerasModel(inputs=input_layer, outputs=output_layer, name="Encoder")

    def compile_decoder_half(self, input_layer, predictor_half):
        layer = input_layer

        layers = self.predictor_layers if predictor_half else self.reconstructor_layers
        for i in range(self.depth):
            use_dropout = i < (self.depth - 1)
            layer = layers[i](layer, use_dropout)

        transition_layer_class = conv_type["conv_block"][False]
        layer = transition_layer_class(filters=self.channels_count, kernel_size=1,
                                       kernel_initializer=self.weights_initializer)(layer)

        output_layer = self.get_activation(self.output_activation)(layer)
        return output_layer

    def compile_decoder(self):
        self.decoder_input_layer = Input(self.compute_decoder_input_shape())

        self.reconstructor_output = self.compile_decoder_half(self.decoder_input_layer, False)
        self.predictor_output = self.compile_decoder_half(self.decoder_input_layer, True)

        output_layer = Concatenate(axis=1)([self.reconstructor_output, self.predictor_output])

        self._decoder = KerasModel(inputs=self.decoder_input_layer, outputs=output_layer, name="Decoder")

    @staticmethod
    def get_activation(activation_config: dict):
        if activation_config["name"] == "leaky_relu":
            return LeakyReLU(alpha=activation_config["alpha"])
        else:
            return Activation(activation_config["name"])

    def get_reconstruction_loss(self, reconstruction_metric_name: str):
        if self.temporal_loss_weights is None:
            reconstruction_loss_weights = np.ones([self.input_sequence_length], dtype=np.float32)
            stop = 0.2
            step = (stop - 1) / self.input_sequence_length
            prediction_loss_weights = np.arange(start=1.0, stop=stop, step=step,
                                                dtype=np.float32)
            loss_weights = np.concatenate([reconstruction_loss_weights, prediction_loss_weights])
            self.temporal_loss_weights = tf.constant(loss_weights)

        def loss_function(y_true, y_pred):
            metric_function = metrics_dict[reconstruction_metric_name]
            reduction_axis = list(range(2, len(y_true.shape)))
            loss = metric_function(y_true, y_pred, axis=reduction_axis)
            loss = loss * self.temporal_loss_weights
            loss = tf.reduce_mean(loss)
            return loss

        return loss_function

    # endregion

    # region Model(s) (Getters)
    def get_true_outputs_placeholder(self) -> tf.Tensor:
        if self._true_outputs_placeholder is None:
            pred_output = self.autoencoder.output
            self._true_outputs_placeholder = tf.placeholder(dtype=pred_output.dtype,
                                                            shape=[None, *self.output_shape],
                                                            name="true_outputs_placeholder")

        return self._true_outputs_placeholder

    def compute_conv_depth(self):
        models = self.autoencoder.layers

        depth = 0
        for model in models:
            if isinstance(model, KerasModel):
                for layer in model.layers:
                    if isinstance(layer, Conv3D):
                        depth += 1
                    elif isinstance(layer, ResBlock3D) or isinstance(layer, ResBlock3DTranspose):
                        depth += len(layer.conv_layers)
                        if layer.projection_layer is not None:
                            depth += 1
                    elif isinstance(layer, DenseBlock3D):
                        depth += layer.depth
                        if layer.transition_layer is not None:
                            depth += 1
        return depth

    @property
    def autoencoder(self):
        if self._autoencoder is None:
            self.compile()
        return self._autoencoder

    @property
    def encoder(self):
        if self._encoder is None:
            self.compile_encoder()
        return self._encoder

    @property
    def decoder(self):
        if self._decoder is None:
            self.compile_decoder()
        return self._decoder

    # endregion

    # region Training
    def train(self,
              database: Database,
              epoch_length: int,
              batch_size: int = 64,
              epochs: int = 1):
        assert isinstance(batch_size, int) or isinstance(batch_size, list)
        assert isinstance(epochs, int) or isinstance(epochs, list)

        if not self.layers_built:
            self.build_layers()

        if self.log_dir is not None:
            return
        self.log_dir = self.__class__.make_log_dir(database)
        print("Logs directory : '{}'".format(self.log_dir))

        self.save_models_info(self.log_dir)

        callbacks = self.build_callbacks(database)
        samples_count = database.train_dataset.samples_count
        callbacks = self.setup_callbacks(callbacks, batch_size, epochs, epoch_length, samples_count)

        callbacks.on_train_begin()
        try:
            self.train_loop(database, callbacks, batch_size, epoch_length, epochs)
        except KeyboardInterrupt:
            print("\n==== Training was stopped by a Keyboard Interrupt ====\n")
        callbacks.on_train_end()

        print("============== Saving models weights... ==============")
        self.save_weights()
        print("======================= Done ! =======================")

        predictions, labels = self.predict_anomalies(database, False)
        self.evaluate_predictions(predictions, labels, os.path.join(self.log_dir, "anomaly_score.png"))

    def resize_database(self, database: Database) -> Database:
        database = database.resized(self.input_image_size, self.input_sequence_length, self.output_sequence_length)
        return database

    def train_loop(self, database: Database, callbacks: CallbackList, batch_size: int, epoch_length: int, epochs: int):
        database = self.resize_database(database)
        database.train_dataset.epoch_length = epoch_length
        database.train_dataset.batch_size = batch_size
        database.test_dataset.batch_size = batch_size

        for _ in range(epochs):
            self.train_epoch(database, callbacks)

    def train_epoch(self, database: Database, callbacks: CallbackList = None):
        epoch_length = len(database.train_dataset)

        callbacks.on_epoch_begin(self.epochs_seen)

        for batch_index in range(epoch_length):
            x, y = database.train_dataset[0]

            # for k in range(len(y)):
            #     for l in range(len(y[0])):
            #         cv2.imshow("y", y[k][l])
            #         cv2.imshow("x", x[k][l % 16])
            #         cv2.waitKey(50)

            batch_logs = {"batch": batch_index, "size": x.shape[0]}
            callbacks.on_batch_begin(batch_index, batch_logs)

            results = self.autoencoder.train_on_batch(x=x, y=y)

            if "metrics" in self.config:
                batch_logs["loss"] = results[0]
                for metric_name, result in zip(self.config["metrics"], results[1:]):
                    batch_logs[metric_name] = result
            else:
                batch_logs["loss"] = results

            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(database, callbacks)

    def on_epoch_end(self,
                     database: Database,
                     callbacks: CallbackList = None,
                     epoch_logs: dict = None):
        if epoch_logs is None:
            epoch_logs = {}

        out_labels = self.autoencoder.metrics_names
        val_outs = self.autoencoder.evaluate_generator(database.test_dataset)
        val_outs = to_list(val_outs)
        for label, val_out in zip(out_labels, val_outs):
            epoch_logs["val_{0}".format(label)] = val_out

        database.on_epoch_end()

        if callbacks:
            callbacks.on_epoch_end(self.epochs_seen, epoch_logs)
        self.epochs_seen += 1

    def save_weights(self):
        self._save_model_weights(self.encoder, "encoder")
        self._save_model_weights(self.decoder, "decoder")

    def _save_model_weights(self, model: KerasModel, name):
        filepath = "{}/weights_{}.h5".format(self.log_dir, name)
        model.save_weights(filepath)

    def load_weights(self, base_filepath):
        AutoEncoderBaseModel._load_model_weights(self.encoder, base_filepath, "encoder")
        AutoEncoderBaseModel._load_model_weights(self.decoder, base_filepath, "decoder")

    @staticmethod
    def _load_model_weights(model: KerasModel, base_filepath, name):
        filepath = "{}/weights_{}.h5".format(base_filepath, name)
        model.load_weights(filepath)

    # endregion

    # region Testing

    def autoencode_video(self,
                         dataset: Dataset,
                         video_index: int,
                         output_video_filepath: str,
                         fps=25.0):
        video_length = dataset.get_video_length(video_index)
        window_length = self.input_sequence_length

        frame_size = tuple(self.output_image_size)
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        video_writer = cv2.VideoWriter(output_video_filepath, fourcc, fps, frame_size)

        for i in tqdm(range(video_length - window_length)):
            input_shard = dataset.get_video_frames(video_index, i, i + window_length)
            input_shard = np.expand_dims(input_shard, axis=0)
            predicted_shard = self.autoencoder.predict_on_batch(input_shard)

            output_frame = predicted_shard[0][1]
            output_frame = (output_frame * 255).astype(np.uint8)
            output_frame = np.repeat(output_frame, 3, axis=-1)
            video_writer.write(output_frame)

        video_writer.release()

    def get_anomalies_raw_predictions_model(self,
                                            include_predictor: bool,
                                            reduction_axis=(2, 3, 4)) -> CallbackModel:
        pred_output: tf.Tensor = self.autoencoder.get_output_at(0)
        if not include_predictor:
            pred_output = pred_output[:, :self.input_sequence_length, ...]
        true_output = tf.placeholder(dtype=pred_output.dtype, shape=pred_output.shape)
        predictions_op = tf.square(pred_output - true_output)
        predictions_op = tf.reduce_sum(predictions_op, axis=reduction_axis)

        if include_predictor:
            predictions_op = self.temporal_loss_weights * predictions_op

        inputs = [self.autoencoder.get_input_at(0), true_output]
        outputs = [predictions_op]
        raw_predictions_model = CallbackModel(inputs, outputs)
        return raw_predictions_model

    def predict_anomalies(self,
                          database: Database,
                          include_predictor: bool,
                          stride=1,
                          normalize_predictions=True):

        dataset = database.test_dataset
        predictions, labels = [], []
        for video_index in range(dataset.videos_count):
            video_length = dataset.get_video_length(video_index)
            window_length = self.output_sequence_length if include_predictor else self.input_sequence_length
            steps_count = 1 + (video_length - window_length) // stride

            video_predictions = np.zeros(shape=[video_length])
            video_labels = np.zeros(shape=[video_length], dtype=np.bool)

            raw_predictions_model = self.get_anomalies_raw_predictions_model(include_predictor)

            for i in range(steps_count):
                start = i * stride
                end = start + window_length

                step_video = dataset.get_video_frames(video_index, start, end)

                step_video = np.expand_dims(step_video, axis=0)
                step_labels = dataset.get_video_frame_labels(video_index, start, end)

                if include_predictor:
                    input_video, output_video = dataset.divide_batch_io(step_video)
                else:
                    input_video = output_video = step_video

                step_predictions = raw_predictions_model.predict(x=[input_video, output_video], batch_size=1)
                step_predictions = np.squeeze(step_predictions)

                video_predictions[start:end] += step_predictions
                if i == 0:
                    video_labels[start:end] = step_labels
                else:
                    video_labels[end - 1] = step_labels[-1]

            predictions_counts = np.ones(shape=[video_length + (1 - window_length) * 2]) * window_length
            predictions_counts = np.concatenate([np.arange(1, window_length),
                                                 predictions_counts,
                                                 np.arange(window_length - 1, 0, -1)])
            video_predictions = video_predictions / predictions_counts

            if include_predictor:
                delta = self.input_sequence_length - self.output_sequence_length
                video_predictions = video_predictions[:delta]
                video_labels = video_labels[:delta]

            predictions.append(video_predictions)
            labels.append(video_labels)

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        return predictions, labels

    def evaluate_predictions(self,
                             predictions: np.ndarray,
                             labels: np.ndarray,
                             output_figure_filepath: str = None):
        if output_figure_filepath is not None:
            plt.plot(predictions)
            plt.plot(labels)
            plt.savefig(output_figure_filepath)

        if self._auc_predictions_placeholder is None:
            placeholder_shape = [None, *predictions.shape[1:]]
            self._auc_predictions_placeholder = tf.placeholder(dtype=tf.float32, shape=placeholder_shape)
            self._auc_labels_placeholder = tf.placeholder(dtype=tf.bool, shape=placeholder_shape)
            roc_op = tf.metrics.auc(self._auc_labels_placeholder, self._auc_predictions_placeholder, curve="ROC",
                                    summation_method="careful_interpolation")
            pr_op = tf.metrics.auc(self._auc_labels_placeholder, self._auc_predictions_placeholder, curve="PR",
                                   summation_method="careful_interpolation")
            self._auc_ops = [roc_op, pr_op]

        session: tf.Session = get_session()
        session.run(tf.local_variables_initializer())
        (_, roc), (_, pr) = session.run(self._auc_ops, feed_dict={self._auc_predictions_placeholder: predictions,
                                                                  self._auc_labels_placeholder: labels})
        print("ROC : {} | PR : {}".format(roc, pr))
        return roc, pr

    # endregion

    # region Callbacks
    def build_callbacks(self, database: Database):
        return self.build_common_callbacks() + self.build_anomaly_callbacks(database)

    def build_common_callbacks(self) -> List[Callback]:
        assert self.tensorboard is None
        self.tensorboard = TensorBoard(log_dir=self.log_dir, update_freq=128)

        base_logger = BaseLogger()
        progbar_logger = ProgbarLogger(count_mode="steps")

        common_callbacks = [base_logger, self.tensorboard, progbar_logger]

        if ("lr_drop_epochs" in self.config) and (self.config["lr_drop_epochs"] > 0):
            lr_scheduler = LearningRateScheduler(self.get_learning_rate_schedule(), verbose=1)
            common_callbacks.append(lr_scheduler)

        return common_callbacks

    def build_anomaly_callbacks(self, database: Database) -> List[Callback]:
        # region Getting database/datasets
        database = self.resize_database(database)
        test_dataset = database.test_dataset
        train_dataset = database.train_dataset
        # endregion

        train_image_callback = self.build_auto_encoding_callback(train_dataset, "train", self.tensorboard)
        test_image_callback = self.build_auto_encoding_callback(test_dataset, "test", self.tensorboard)

        # region Getting samples used for AUC callbacks
        samples = test_dataset.sample(batch_size=512, seed=16, sampled_videos_count=8, return_labels=True)
        videos, frame_labels, pixel_labels = samples
        videos = test_dataset.divide_batch_io(videos)
        # endregion

        # region Frame level error AUC (ROC)
        frame_predictions_model = self.build_frame_level_error_callback_model()
        frame_auc_callback = AUCCallback(frame_predictions_model, self.tensorboard,
                                         videos, frame_labels,
                                         plot_size=(128, 128), batch_size=32,
                                         name="Frame_Level_Error_AUC", epoch_freq=5)
        # endregion

        anomaly_callbacks = [train_image_callback, test_image_callback, frame_auc_callback]

        # region Pixel level error AUC (ROC)
        # TODO : Check labels size in UCSDDatabase
        # if pixel_labels is not None:
        #     pixel_auc_callback = AUCCallback(pixel_predictions_model, self.tensorboard,
        #                                      videos, pixel_labels,
        #                                      plot_size=(128, 128), batch_size=32,
        #                                      num_thresholds=20, name="Pixel_Level_Error_AUC", epoch_freq=5)
        #     anomaly_callbacks.append(pixel_auc_callback)
        # endregion

        return anomaly_callbacks

    def setup_callbacks(self,
                        callbacks: CallbackList or List[Callback],
                        batch_size: int,
                        epochs: int,
                        epoch_length: int,
                        samples_count: int) -> CallbackList:
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self.autoencoder)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': epoch_length,
            'samples': samples_count,
            'verbose': 1,
            'do_validation': True,
            'metrics': self.callback_metrics(),
        })
        return callbacks

    def callback_metrics(self):
        metrics_names = self.autoencoder.metrics_names
        validation_callbacks = ["val_{0}".format(name) for name in metrics_names]
        callback_metrics = copy.copy(metrics_names) + validation_callbacks
        return callback_metrics

    def build_auto_encoding_callback(self,
                                     dataset: Dataset,
                                     name: str,
                                     tensorboard: TensorBoard,
                                     frequency="epoch") -> ImageCallback:

        videos = dataset.get_batch(self.image_summaries_max_outputs, seed=1, apply_preprocess_step=False,
                                   max_shard_count=self.image_summaries_max_outputs)

        true_outputs_placeholder = self.get_true_outputs_placeholder()
        summary_inputs = [self.autoencoder.input, true_outputs_placeholder]

        true_outputs = self.normalize_image_tensor(true_outputs_placeholder)
        pred_outputs = self.normalize_image_tensor(self.autoencoder.output)

        io_delta = (pred_outputs - true_outputs) * (tf.cast(pred_outputs < true_outputs, dtype=tf.uint8) * 254 + 1)

        with tf.name_scope("composite_error_computation"):
            error = tf.cast(io_delta, tf.float32) / 255.0
            composite_hue = tf.multiply(1.0 - error, 0.25, name="composite_hue")
            composite_saturation = tf.identity(error, name="composite_saturation")
            composite_value = tf.cast(true_outputs, tf.float32) / 255.0
            composite_value = tf.maximum(composite_value, error, name="composite_value")
            composite_hsv = tf.concat([composite_hue, composite_saturation, composite_value], axis=-1,
                                      name="composite_hsv")
            composite_rgb = tf.image.hsv_to_rgb(composite_hsv)
            composite_rgb = tf.cast(composite_rgb * 255.0, tf.uint8, name="composite_rgb")

        max_outputs = self.image_summaries_max_outputs
        one_shot_summaries = [image_summary(name + "_true_outputs", true_outputs, max_outputs, fps=8)]
        repeated_summaries = [image_summary(name + "_pred_outputs", pred_outputs, max_outputs, fps=8),
                              image_summary(name + "_delta", io_delta, max_outputs, fps=8),
                              image_summary(name + "_composite", composite_rgb, max_outputs, fps=8)]

        one_shot_summary_model = CallbackModel(summary_inputs, one_shot_summaries, output_is_summary=True)
        repeated_summary_model = CallbackModel(summary_inputs, repeated_summaries, output_is_summary=True)

        return ImageCallback(one_shot_summary_model, repeated_summary_model, videos, tensorboard, frequency)

    def normalize_image_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("normalize_image_tensor"):
            normalized = (tensor - self._min_output_constant) * self._inv_output_range_constant
            normalized = tf.cast(normalized * uint8_max_constant, tf.uint8)
        return normalized

    def build_pixel_level_error_callback_model(self):
        true_outputs = self.get_true_outputs_placeholder()

        error = tf.square(self.autoencoder.output - true_outputs)

        pixel_predictions_model = CallbackModel([self.autoencoder.input, true_outputs], error)

        return pixel_predictions_model

    def build_frame_level_error_callback_model(self):
        true_outputs = self.get_true_outputs_placeholder()

        squared_delta = tf.square(self.autoencoder.output - true_outputs)
        average_error = tf.reduce_sum(squared_delta, axis=[2, 3, 4])

        frame_predictions_model = CallbackModel([self.autoencoder.input, true_outputs], average_error)

        return frame_predictions_model

    @staticmethod
    def update_callbacks_param(callbacks: CallbackList,
                               param_name: str,
                               param_value: Any):
        for callback in callbacks:
            callback.params[param_name] = param_value
            if hasattr(callback, param_name):
                setattr(callback, param_name, param_value)

    def get_learning_rate_schedule(self):
        lr_drop_epochs = self.config["lr_drop_epochs"]

        def schedule(epoch, learning_rate):
            if (epoch % lr_drop_epochs) == (lr_drop_epochs - 1):
                return learning_rate * 0.5
            else:
                return learning_rate

        return schedule

    # endregion

    # region Log dir

    @classmethod
    def make_log_dir(cls,
                     database: Database):
        project_log_dir = "../logs/AutoEncoding-Anomalies"
        base_dir = os.path.join(project_log_dir, database.__class__.__name__, cls.__name__)
        log_dir = get_log_dir(base_dir)
        return log_dir

    def save_models_info(self, log_dir: str):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        AutoEncoderBaseModel.save_model_info(log_dir, self.encoder)
        AutoEncoderBaseModel.save_model_info(log_dir, self.decoder)
        AutoEncoderBaseModel.save_model_info(log_dir, self.autoencoder)

        config_filename = os.path.join(log_dir, "global_config.json")
        with open(config_filename, "w") as file:
            json.dump(self.config, file)

    @staticmethod
    def save_model_info(log_dir: str, model: KerasModel):
        keras_config_filename = os.path.join(log_dir, "{}_keras_config.json".format(model.name))
        with open(keras_config_filename, "w") as file:
            file.write(model.to_json())

        summary_filename = os.path.join(log_dir, "{}_summary.txt".format(model.name))
        with open(summary_filename, "w") as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

    # endregion

    # region Properties
    @property
    def input_image_size(self):
        return self.input_shape[1:-1]

    @property
    def input_sequence_length(self):
        return self.input_shape[0]

    @property
    def output_image_size(self):
        return self.output_shape[1:-1]

    @property
    def output_sequence_length(self):
        return self.output_shape[0]
    # endregion
