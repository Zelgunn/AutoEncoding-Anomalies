from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Activation, LeakyReLU, Dense, Dropout, Input
from tensorflow.python.keras.layers import Layer, Reshape, Concatenate, Lambda
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, AveragePooling3D, UpSampling3D
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import TensorBoard, CallbackList, Callback, ProgbarLogger, BaseLogger
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.backend import binary_crossentropy, set_learning_phase

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
import json
import copy
from typing import List, Union, Dict, Tuple, Optional

from layers import ResBlock3D, ResBlock3DTranspose, DenseBlock3D, SpectralNormalization
from layers.utility_layers import RawPredictionsLayer
from datasets import SubsetLoader, DatasetLoader
from modalities import RawVideo
from callbacks import ImageCallback, MultipleModelsCheckpoint
from utils.train_utils import get_log_dir
from utils.misc_utils import to_list


# region Containers
class LayerBlock(object):
    def __init__(self,
                 conv_layer: Layer,
                 pool_layer: Optional[Layer],
                 upsampling_layer: Optional[Layer],
                 activation_layer: Optional[Layer],
                 dropout_layer: Optional[Dropout],
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

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        if self.upsampling_layer is not None:
            input_shape: tf.TensorShape = self.upsampling_layer.compute_output_shape(input_shape)

        input_shape: tf.TensorShape = self.conv.compute_output_shape(input_shape)

        if self.pooling is not None:
            input_shape: tf.TensorShape = self.pooling.compute_output_shape(input_shape)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        return tuple(input_shape)


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

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        for layer in self.layers:
            input_shape: Tuple = layer.compute_output_shape(input_shape)
        return input_shape


# endregion

# region Output activations

output_activation_ranges = {"sigmoid": (0.0, 1.0),
                            "tanh": (-1.0, 1.0)}


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

conv_type = {"conv_block": {False: Conv3D, True: Conv3DTranspose},
             "residual_block": {False: ResBlock3D, True: ResBlock3DTranspose},
             "dense_block": {False: DenseBlock3D}}

pool_type = {"max": MaxPooling3D,
             "average": AveragePooling3D}
# endregion

uint8_max_constant = tf.constant(255.0, dtype=tf.float32)


class AutoEncoderBaseModel(ABC):
    # region Initialization
    def __init__(self):
        # Fuse with Load config (to reduce amount of None in __init__)
        self.image_summaries_max_outputs = 3

        self.layers_built = False
        self.embeddings_layer: Optional[Union[Layer, LayerStack]] = None
        self.encoder_layers: List[LayerStack] = []
        self.reconstructor_layers: List[LayerStack] = []
        self.predictor_layers: List[LayerStack] = []

        self.encoder_input_layer: Input = None
        self.decoder_input_layer: Input = None

        self._encoder: Optional[KerasModel] = None
        self._decoder: Optional[KerasModel] = None
        self._autoencoder: Optional[KerasModel] = None
        self._raw_predictions_models: Dict[bool, Dict[Tuple, KerasModel]] = {False: {}, True: {}}
        self._reconstructor_model: Optional[KerasModel] = None

        self.decoder_input_layer = None
        self.reconstructor_output = None
        self.predictor_output = None

        self._train_dataset_iterator = None
        self._test_dataset_iterator = None

        self.config: Optional[dict] = None
        self.encoder_depth: Optional[int] = None
        self.decoder_depth: Optional[int] = None

        self.channels_first = True
        self.input_shape: Optional[List[int]] = None
        self.output_shape: Optional[List[int]] = None
        self.channels_count: Optional[int] = None

        self.embeddings_size = None
        self.use_dense_embeddings = None

        self._true_outputs_placeholder: Optional[tf.Tensor] = None
        self._auc_predictions_placeholder: Optional[tf.Tensor] = None
        self._auc_labels_placeholder: Optional[tf.Tensor] = None
        self._auc_ops: Optional[List[tf.Tensor, tf.Tensor]] = None
        self._auc_summary_ops = None

        self.default_activation = None
        self.embeddings_activation = None
        self.output_activation = None

        self.weights_initializer = None

        self.block_type_name = None
        self.pooling = None
        self.upsampling = None
        self.use_batch_normalization = None

        self._temporal_loss_weights = None
        self.optimizer = None

        self.log_dir = None
        self.tensorboard: Optional[tf.python.keras.callbacks.TensorBoard] = None

        self.epochs_seen = 0
        self.output_range = None
        self._min_output_constant: Optional[tf.Tensor] = None
        self._max_output_constant: Optional[tf.Tensor] = None
        self._inv_output_range_constant: Optional[tf.Tensor] = None
        self.use_spectral_norm = False
        self.weight_decay_regularizer = None

        self.train_data_augmentations = None
        self.test_data_augmentations = None

    def load_config(self, config_file: str, alt_config_file: str):
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
        self.embeddings_size = self.config["embeddings_layer"]["filters"]
        self.use_dense_embeddings = self.config["use_dense_embeddings"] == "True"
        # endregion

        self.encoder_depth = len(self.config["encoder"])
        self.decoder_depth = len(self.config["decoder"])
        self._true_outputs_placeholder = None

        # region I/O shapes
        if "data_format" in self.config:
            data_format = self.config["data_format"]
            assert data_format in ["channels_first",
                                   "channels_last"], "data_format `{}` not in [channels_first, channels_last]"
            self.channels_first = data_format == "channels_first"

        self.input_shape = tuple(self.config["input_shape"])
        if self.channels_first:
            self.input_shape = (self.input_shape[-1], *self.input_shape[:-1])
        if self.channels_first:
            self.output_shape = (self.input_shape[0], self.input_shape[1] * 2, *self.input_shape[2:])
        else:
            self.output_shape = (self.input_shape[0] * 2, *self.input_shape[1:])
        self.channels_count = self.input_shape[self.channels_axis]
        # endregion

        # region Activations
        self.default_activation = self.config["default_activation"]
        self.embeddings_activation = self.config["embeddings_activation"]
        self.output_activation = self.config["output_activation"]

        self.output_range = output_activation_ranges[self.output_activation["name"]]
        self._min_output_constant = tf.constant(self.output_range[0], name="min_output")
        self._max_output_constant = tf.constant(self.output_range[1], name="max_output")
        inv_range = 1.0 / (self.output_range[1] - self.output_range[0])
        self._inv_output_range_constant = tf.constant(inv_range, name="inv_output_range")
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

        # self.train_data_augmentations = self.build_data_augmentations(True)
        # self.test_data_augmentations = self.build_data_augmentations(False)

        # region Seed
        if "seed" in self.config:
            seed = self.config["seed"]
        else:
            seed = np.random.randint(2147483647)
            self.config["seed"] = seed
        tf.random.set_seed(seed)
        # endregion

    # region Data Augmentation
    # TODO : Move Data Augmentation from AutoEncoderBaseModel to Datasets
    # def build_data_augmentations(self, train: bool):
    #     data_preprocessors = []
    #
    #     section_name = "train" if train else "test"
    #     section = self.config["data_generators"][section_name]
    #
    #     if "cropping" in section:
    #         cropping_section = section["cropping"]
    #         width_range = cropping_section["width_range"]
    #         height_range = cropping_section["height_range"] if "height_range" in cropping_section else None
    #         keep_ratio = (cropping_section["keep_ratio"] == "True") if "keep_ratio" in cropping_section else True
    #         random_cropper = RandomCropper(width_range, height_range, keep_ratio)
    #         data_preprocessors.append(random_cropper)
    #
    #     if "blurring" in section:
    #         blurring_section = section["blurring"]
    #         max_sigma = blurring_section["max_sigma"] if "max_sigma" in blurring_section else 5.0
    #         kernel_size = tuple(blurring_section["kernel_size"]) if "kernel_size" in blurring_section else (3, 3)
    #         random_blurrer = GaussianBlurrer(max_sigma, kernel_size, apply_on_outputs=False)
    #         data_preprocessors.append(random_blurrer)
    #
    #     if "brightness" in section:
    #         brightness_section = section["brightness"]
    #         gain = brightness_section["gain"] if "gain" in brightness_section else None
    #         bias = brightness_section["bias"] if "bias" in brightness_section else None
    #         brightness_shifter = BrightnessShifter(gain=gain, bias=bias, values_range=self.output_range)
    #         data_preprocessors.append(brightness_shifter)
    #
    #     if "dropout_rate" in section:
    #         dropout_noise_rate = section["dropout_rate"]
    #         dropout_noiser = DropoutNoiser(inputs_dropout_rate=dropout_noise_rate)
    #         data_preprocessors.append(dropout_noiser)
    #
    #     return data_preprocessors

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
            kernel_size = self.config["embeddings_layer"]["kernel_size"]
            self.embeddings_layer = conv(filters=self.embeddings_size,
                                         kernel_size=kernel_size,
                                         padding="same",
                                         kernel_initializer=self.weights_initializer,
                                         kernel_regularizer=self.weight_decay_regularizer,
                                         bias_regularizer=self.weight_decay_regularizer,
                                         data_format=self.data_format,
                                         name="embeddings_layer")

        for stack_info in self.config["decoder"]:
            stack = self.build_deconv_stack(stack_info)
            self.reconstructor_layers.append(stack)
            stack = self.build_deconv_stack(stack_info)
            self.predictor_layers.append(stack)
        self.layers_built = True

    def build_conv_stack(self, stack_info: dict, transpose=False) -> LayerStack:
        if self.block_type_name == "dense_block":
            return self.build_dense_block(stack_info, transpose)
        elif self.block_type_name == "residual_block":
            return self.build_residual_block(stack_info, transpose)
        else:
            return self.build_layer_stack(stack_info, transpose)

    def build_deconv_stack(self, stack_info: dict) -> LayerStack:
        return self.build_conv_stack(stack_info, transpose=True)

    def build_residual_block(self, stack_info: dict, transpose):
        depth: int = stack_info["depth"]
        stack = LayerStack()

        strides = stack_info["strides"]
        upsampling_layer, pool_layer, conv_strides = None, None, 1
        if transpose:
            if self.upsampling == "strides":
                conv_strides = strides
            else:
                upsampling_layer = UpSampling3D(size=strides)
        else:
            if self.pooling == "strides":
                conv_strides = strides
            else:
                pool_layer = pool_type[self.pooling](pool_size=strides)

        conv_block_type = self.get_block_type(transpose)
        activation = AutoEncoderBaseModel.get_activation(self.default_activation)

        conv_layer = conv_block_type(filters=stack_info["filters"],
                                     basic_block_count=depth,
                                     basic_block_depth=2,
                                     kernel_size=stack_info["kernel_size"],
                                     strides=conv_strides,
                                     activation=activation,
                                     use_bias=True,
                                     kernel_initializer=self.weights_initializer,
                                     kernel_regularizer=self.weight_decay_regularizer,
                                     bias_regularizer=self.weight_decay_regularizer
                                     )

        if self.use_spectral_norm:
            conv_layer = SpectralNormalization(conv_layer)

        dropout = Dropout(rate=stack_info["dropout"]) if "dropout" in stack_info else None

        layer_block = LayerBlock(conv_layer=conv_layer,
                                 pool_layer=pool_layer,
                                 upsampling_layer=upsampling_layer,
                                 activation_layer=None,
                                 dropout_layer=dropout,
                                 is_transpose=transpose,
                                 add_batch_normalization=self.use_batch_normalization)
        stack.add_layer(layer_block)

        return stack

    def build_dense_block(self, stack_info: dict, transpose) -> LayerStack:
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
        stack_strides = stack_info["strides"] if "strides" in stack_info else 1
        if isinstance(stack_strides, int):
            stack_strides_is_one = stack_strides == 1
        else:
            stack_strides_is_one = all([stride == 1 for stride in stack_strides])

        stack = LayerStack()
        for i in range(depth):
            # region Pooling/Upsampling layer
            if (i < (depth - 1)) or stack_strides_is_one:
                conv_layer_strides = 1
                pool_layer = None
                upsampling_layer = None
            elif transpose:
                pool_layer = None
                if self.upsampling == "strides":
                    conv_layer_strides = stack_strides
                    upsampling_layer = None
                else:
                    conv_layer_strides = 1
                    upsampling_layer = UpSampling3D(size=stack_strides)
            else:
                upsampling_layer = None
                if self.pooling == "strides":
                    conv_layer_strides = stack_strides
                    pool_layer = None
                else:
                    conv_layer_strides = 1
                    pool_layer = pool_type[self.pooling](pool_size=stack_strides)
            # endregion

            # region Convolutional layer
            conv_layer_kwargs = {"kernel_size": stack_info["kernel_size"],
                                 "filters": stack_info["filters"],
                                 "strides": conv_layer_strides,
                                 "kernel_initializer": self.weights_initializer,
                                 "kernel_regularizer": self.weight_decay_regularizer,
                                 "bias_regularizer": self.weight_decay_regularizer,
                                 "data_format": self.data_format}

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

    def compute_embeddings_input_shape(self) -> Tuple:
        embeddings_shape = (None, *self.input_shape)

        for stack in self.encoder_layers:
            embeddings_shape: tuple = stack.compute_output_shape(embeddings_shape)

        return embeddings_shape[1:]

    def compute_embeddings_output_shape(self) -> tuple:
        embeddings_shape: tuple = self.compute_embeddings_input_shape()
        if self.use_dense_embeddings:
            embeddings_shape = embeddings_shape[:-1]
            embeddings_shape_prod = np.prod(embeddings_shape)
            assert (self.embeddings_size % embeddings_shape_prod) == 0, \
                "embeddings_size = {}, embeddings_shape_prod = {}".format(self.embeddings_size, embeddings_shape_prod)
            embeddings_filters = self.embeddings_size // embeddings_shape_prod
            embeddings_shape = (*embeddings_shape, embeddings_filters)
        else:
            embeddings_shape = (None, *embeddings_shape)
            embeddings_tensor_shape = self.embeddings_layer.compute_output_shape(embeddings_shape)
            embeddings_tensor_shape = embeddings_tensor_shape[1:]
            if isinstance(embeddings_tensor_shape, tf.TensorShape):
                embeddings_tensor_shape = embeddings_tensor_shape.as_list()
            embeddings_shape = tuple(embeddings_tensor_shape)
        return embeddings_shape

    def compute_decoder_input_shape(self) -> tuple:
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
        self.encoder_input_layer = Input(self.input_shape, name="encoder_input_layer")
        layer = self.encoder_input_layer

        for i in range(self.encoder_depth):
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

        self._encoder = KerasModel(inputs=self.encoder_input_layer,
                                   outputs=output_layer,
                                   name="Encoder")

    def compile_decoder_half(self, predictor_half):
        layer = self.decoder_input_layer

        layers = self.predictor_layers if predictor_half else self.reconstructor_layers
        for i in range(self.decoder_depth):
            use_dropout = i < (self.decoder_depth - 1)
            layer = layers[i](layer, use_dropout)

        transition_layer_class = conv_type["conv_block"][False]
        layer = transition_layer_class(filters=self.channels_count, kernel_size=1,
                                       kernel_initializer=self.weights_initializer,
                                       data_format=self.data_format)(layer)

        output_layer = self.get_activation(self.output_activation)(layer)
        return output_layer

    def compile_decoder(self):
        self.decoder_input_layer = Input(self.compute_decoder_input_shape(), name="decoder_input_layer")

        self.reconstructor_output = self.compile_decoder_half(False)
        self.predictor_output = self.compile_decoder_half(True)

        output_layer = Concatenate(axis=self.time_axis)([self.reconstructor_output, self.predictor_output])

        self._decoder = KerasModel(inputs=self.decoder_input_layer, outputs=output_layer, name="Decoder")

    @staticmethod
    def get_activation(activation_config: dict):
        if activation_config["name"] == "leaky_relu":
            return LeakyReLU(alpha=activation_config["alpha"])
        else:
            return Activation(activation_config["name"])

    @property
    def temporal_loss_weights(self):
        if self._temporal_loss_weights is None:
            reconstruction_loss_weights = np.ones([self.input_sequence_length], dtype=np.float32)
            stop = 0.2
            step = (stop - 1) / self.input_sequence_length
            prediction_loss_weights = np.arange(start=1.0, stop=stop, step=step, dtype=np.float32)
            loss_weights = np.concatenate([reconstruction_loss_weights, prediction_loss_weights])
            self._temporal_loss_weights = tf.constant(loss_weights, name="temporal_loss_weights")
        return self._temporal_loss_weights

    def get_reconstruction_loss(self, reconstruction_metric_name: str):
        def loss_function(y_true, y_pred):
            metric_function = metrics_dict[reconstruction_metric_name]
            reduction_axis = list(range(1, len(y_true.shape)))
            reduction_axis.remove(self.time_axis)
            loss = metric_function(y_true, y_pred, axis=reduction_axis)
            loss = loss * self.temporal_loss_weights
            loss = tf.reduce_mean(loss)
            return loss

        return loss_function

    # endregion

    # region Model(s) (Getters)
    def compute_conv_depth(self):
        models = self.autoencoder.layers

        depth = 0
        for model in models:
            if isinstance(model, KerasModel):
                for layer in model.layers:
                    if isinstance(layer, Conv3D):
                        depth += 1
                    elif isinstance(layer, ResBlock3D) or isinstance(layer, ResBlock3DTranspose):
                        depth += layer.basic_block_count * layer.basic_block_depth
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

    @property
    def saved_models(self) -> Dict[str, KerasModel]:
        return {"encoder": self.encoder,
                "decoder": self.decoder}

    # endregion

    # region Training
    def train(self,
              dataset: DatasetLoader,
              dataset_name: str,
              epoch_length: int,
              batch_size: int = 64,
              epochs: int = 1):
        assert isinstance(batch_size, int) or isinstance(batch_size, list)
        assert isinstance(epochs, int) or isinstance(epochs, list)

        self.on_train_begin(dataset, dataset_name, batch_size)

        callbacks = self.build_callbacks(dataset)
        callbacks = self.setup_callbacks(callbacks, batch_size, epochs, epoch_length)
        callbacks.on_train_begin()

        try:
            self.train_loop(dataset, callbacks, batch_size, epoch_length, epochs)

        except KeyboardInterrupt:
            print("\n==== Training was stopped by a Keyboard Interrupt ====\n")

        callbacks.on_train_end()
        self.on_train_end()

    def train_loop(self,
                   dataset: DatasetLoader,
                   callbacks: CallbackList,
                   batch_size: int,
                   epoch_length: int,
                   epochs: int):

        for _ in range(epochs):
            self.train_epoch(dataset, callbacks, batch_size, epoch_length)

    def train_epoch(self,
                    dataset: DatasetLoader,
                    callbacks: CallbackList,
                    batch_size: int,
                    epoch_length: int,
                    ):

        set_learning_phase(1)
        callbacks.on_epoch_begin(self.epochs_seen)

        for batch_index in range(epoch_length):
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            results = self.autoencoder.train_on_batch(self._train_dataset_iterator)

            if "metrics" in self.config and len(self.config["metrics"]) > 0:
                batch_logs["loss"] = results[0]
                for metric_name, result in zip(self.config["metrics"], results[1:]):
                    batch_logs[metric_name] = result
            else:
                batch_logs["loss"] = results

            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(callbacks)

    # region Make dataset iterators
    def make_dataset_iterators(self, dataset: DatasetLoader, batch_size: int):
        if self._train_dataset_iterator is not None:
            return

        self._train_dataset_iterator = self.batch_and_prefetch(dataset.train_subset.tf_dataset, batch_size)
        self._test_dataset_iterator = self.batch_and_prefetch(dataset.test_subset.tf_dataset, batch_size,
                                                              prefetch=False)

    @staticmethod
    def batch_and_prefetch(dataset: tf.data.Dataset, batch_size: int, prefetch=True):
        dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # TODO : Find a better way - this should allow to use several mods but currently we only have one
        dataset = dataset.map(lambda inputs, outputs: (inputs[0], outputs[0]))
        # dataset_iterator = MultiDeviceIterator(dataset_iterator, devices=["/gpu:0"])
        # dataset_iterator = dataset_iterator[0] # Select 1st GPU with [0]

        return dataset

    # endregion

    def on_epoch_end(self,
                     callbacks: CallbackList = None,
                     epoch_logs: dict = None):

        set_learning_phase(0)
        if epoch_logs is None:
            epoch_logs = {}

        out_labels = self.autoencoder.metrics_names

        # TODO : Change validation steps
        val_outs = self.autoencoder.evaluate(self._test_dataset_iterator, steps=64, verbose=0)
        val_outs = to_list(val_outs)
        for label, val_out in zip(out_labels, val_outs):
            epoch_logs["val_{0}".format(label)] = val_out

        if callbacks:
            callbacks.on_epoch_end(self.epochs_seen, epoch_logs)

        self.epochs_seen += 1

        # if self.epochs_seen % 5 == 0:
        #     predictions, labels, lengths = self.predict_anomalies(dataset)
        #     roc, pr = self.evaluate_predictions(predictions, labels, lengths, log_in_tensorboard=True)
        #     print("Epochs seen = {} | ROC = {} | PR = {}".format(self.epochs_seen, roc, pr))

    def on_train_begin(self,
                       dataset: DatasetLoader,
                       dataset_name: str,
                       batch_size: int):
        if not self.layers_built:
            self.build_layers()

        if self.log_dir is not None:
            return
        self.log_dir = self.make_log_dir(dataset_name)
        print("Logs directory : '{}'".format(self.log_dir))

        self.save_models_info(self.log_dir)
        self.make_dataset_iterators(dataset, batch_size)

    def on_train_end(self):
        print("============== Saving models weights... ==============")
        self.save_weights()
        print("======================= Done ! =======================")

    # endregion

    # region Testing
    def autoencode_video(self,
                         subset: SubsetLoader,
                         video_index: int,
                         output_video_filepath: str,
                         fps=25.0,
                         max_frame_count=1000):

        frame_size = tuple(self.output_image_size)
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        video_writer = cv2.VideoWriter(output_video_filepath, fourcc, fps, frame_size)

        source_browser = subset.get_source_browser(video_index, RawVideo, stride=1)
        predicted = self.autoencoder.predict(source_browser, steps=max_frame_count)

        predicted = (predicted * 255).astype(np.uint8)
        predicted = np.repeat(predicted, 3, axis=-1)

        for frame in predicted:
            video_writer.write(frame)

        video_writer.release()

    def get_anomalies_raw_predictions_model(self, reduction_axis=(2, 3, 4), include_labels_io=False) -> KerasModel:
        if reduction_axis not in self._raw_predictions_models:
            reconstructor = KerasModel(inputs=self.decoder_input_layer, outputs=self.reconstructor_output,
                                       name="Reconstructor")
            encoder_output = self.encoder(self.encoder.get_input_at(0))
            if isinstance(encoder_output, list):
                encoder_output = encoder_output[0]
            pred_output = reconstructor(encoder_output)
            true_output = self.encoder.get_input_at(0)
            error = RawPredictionsLayer(reduction_axis)([pred_output, true_output])

            labels_input_layer = Input(shape=[None, 2], dtype=tf.float32, name="labels_input_layer")
            labels_output_layer = Lambda(tf.identity, name="labels_identity")(labels_input_layer)

            inputs = [self.encoder.get_input_at(0)]
            outputs = [error]
            if include_labels_io:
                inputs += [labels_input_layer]
                outputs += [labels_output_layer]

            raw_predictions_model = KerasModel(inputs=inputs, outputs=outputs)

            self._raw_predictions_models[include_labels_io][reduction_axis] = raw_predictions_model

        return self._raw_predictions_models[include_labels_io][reduction_axis]

    # region Using attractors
    def get_reconstructor_run_model(self) -> KerasModel:
        if self._reconstructor_model is None:
            reconstructor = KerasModel(inputs=self.decoder_input_layer, outputs=self.reconstructor_output,
                                       name="Reconstructor")
            encoder_output = self.encoder(self.encoder.get_input_at(0))
            if isinstance(encoder_output, list):
                encoder_output = encoder_output[0]
            pred_output = reconstructor(encoder_output)

            inputs = self.encoder.get_input_at(0)
            outputs = pred_output

            self._reconstructor_model = KerasModel(inputs, outputs)

        return self._reconstructor_model

    def predict_anomalies_on_single_example_until_convergence(self,
                                                              step_video: np.ndarray,
                                                              convergence_threshold: float,
                                                              max_iterations: int
                                                              ):
        reconstructor_run_model = self.get_reconstructor_run_model()
        total_error = None
        # total_error = []

        for i in range(max_iterations):
            reconstructed_video = reconstructor_run_model.predict(x=step_video, batch_size=1)
            reconstructed_video = reconstructed_video[0]
            step_error = np.square(reconstructed_video - step_video)
            step_error = np.mean(step_error, axis=(2, 3, 4))
            step_error = np.squeeze(step_error, axis=0)
            if i == 0:
                total_error = step_error
            else:
                total_error -= step_error
            if step_error.mean() < convergence_threshold:
                break
            step_video = reconstructed_video

        return total_error

    # endregion

    # TODO : Re-implement attractors
    # noinspection PyUnusedLocal
    def predict_anomalies_on_video(self,
                                   subset: SubsetLoader,
                                   video_index: int,
                                   stride: int,
                                   normalize_predictions=False,
                                   convergence_threshold=None,
                                   max_iterations=1):

        raw_predictions_model = self.get_anomalies_raw_predictions_model(include_labels_io=True)
        source_browser = subset.get_source_browser(video_index, RawVideo, stride)
        # TODO : Get steps count
        steps_count = 100000
        predictions, labels = raw_predictions_model.predict(source_browser, steps=steps_count)
        labels = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
        labels = np.any(labels, axis=-1)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / predictions.max()

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    stride: int,
                                    convergence_threshold=None,
                                    max_iterations=1,
                                    max_videos=10):
        predictions, labels = [], []

        video_count = min(max_videos, len(subset.subset_folders))
        for video_index in range(video_count):
            video_results = self.predict_anomalies_on_video(subset, video_index, stride,
                                                            normalize_predictions=True,
                                                            convergence_threshold=convergence_threshold,
                                                            max_iterations=max_iterations)
            video_predictions, video_labels = video_results
            predictions.append(video_predictions)
            labels.append(video_labels)

        return predictions, labels

    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          stride=1,
                          normalize_predictions=True,
                          convergence_threshold=None,
                          max_iterations=1,
                          max_videos=10):
        predictions, labels = self.predict_anomalies_on_subset(dataset.test_subset,
                                                               stride,
                                                               convergence_threshold,
                                                               max_iterations,
                                                               max_videos)

        lengths = np.empty(shape=[len(labels)], dtype=np.int32)
        for i in range(len(labels)):
            lengths[i] = len(labels[i])
            if i > 0:
                lengths[i] += lengths[i - 1]

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        return predictions, labels, lengths

    def evaluate_predictions(self,
                             predictions: np.ndarray,
                             labels: np.ndarray,
                             lengths: np.ndarray = None,
                             output_figure_filepath: str = None,
                             log_in_tensorboard=False):
        if output_figure_filepath is not None:
            plt.plot(predictions, linewidth=0.5)
            plt.plot(labels, alpha=0.75, linewidth=0.5)
            if lengths is not None:
                lengths_splits = np.zeros(shape=predictions.shape, dtype=np.float32)
                lengths_splits[lengths - 1] = 1.0
                plt.plot(lengths_splits, alpha=0.5, linewidth=0.2)
            plt.savefig(output_figure_filepath, dpi=500)
            plt.clf()  # clf = clear figure...

        roc = tf.metrics.AUC(curve="ROC")
        pr = tf.metrics.AUC(curve="PR")

        roc.update_state(labels, predictions)
        pr.update_state(labels, predictions)

        roc_result = roc.result()
        pr_result = pr.result()

        if log_in_tensorboard is not None:
            # noinspection PyProtectedMember
            with self.tensorboard._get_writer(self.tensorboard._train_run_name).as_default():
                tf.summary.scalar(name="Video_ROC_AUC", data=roc_result, step=self.epochs_seen)
                tf.summary.scalar(name="Video_PR_AUC", data=pr_result, step=self.epochs_seen)

        return roc, pr

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             stride=1,
                             convergence_threshold=None,
                             max_iterations=1):
        predictions, labels, lengths = self.predict_anomalies(dataset,
                                                              stride=stride,
                                                              normalize_predictions=True,
                                                              convergence_threshold=convergence_threshold,
                                                              max_iterations=max_iterations)
        graph_filepath = os.path.join(self.log_dir, "Anomaly_score.png")
        roc, pr = self.evaluate_predictions(predictions, labels, lengths, graph_filepath)
        print("Anomaly_score : ROC = {} | PR = {}".format(roc, pr))
        return roc, pr

    # thresholds = [
    #         {"convergence_threshold": None, "max_iterations": 1},
    #         {"convergence_threshold": 2e-4, "max_iterations": 20},
    #         {"convergence_threshold": 1e-4, "max_iterations": 20},
    #     ]

    def evaluate_attractors(self,
                            previous_weights_to_load: str,
                            thresholds: List[Dict[str, Optional[int]]],
                            dataset: DatasetLoader,
                            start_epoch: int,
                            end_epoch: int,
                            step: int,
                            ):
        from tqdm import tqdm

        count = (start_epoch - end_epoch) // step

        roc_plot = np.zeros(shape=[len(thresholds), count])
        pr_plot = np.zeros(shape=[len(thresholds), count])

        def save_attractor_results(limit):
            np.save(previous_weights_to_load + "/roc_plot.npy", roc_plot[:, :limit])
            np.save(previous_weights_to_load + "/pr_plot.npy", pr_plot[:, :limit])

            lines, lines_names = [], []
            for k in range(len(thresholds)):
                line_name = str(thresholds[k])
                line, = plt.plot(roc_plot[k, :limit], label=line_name)

                lines.append(line)
                lines_names.append(line_name)
            plt.legend(lines, lines_names)

            plt.savefig(previous_weights_to_load + "/roc_plot.png", dpi=500)
            plt.clf()

            lines, lines_names = [], []
            for k in range(len(thresholds)):
                line_name = str(thresholds[k])
                line, = plt.plot(pr_plot[k, :limit], label=line_name)

                lines.append(line)
                lines_names.append(line_name)
            plt.legend(lines, lines_names)

            plt.savefig(previous_weights_to_load + "/pr_plot.png", dpi=500)
            plt.clf()

        for i in tqdm(range(count)):
            self.load_weights(previous_weights_to_load, epoch=start_epoch + step * i)
            for j, params in enumerate(thresholds):
                predictions, labels, lengths = self.predict_anomalies(dataset, **params)
                roc, pr = self.evaluate_predictions(predictions, labels, lengths)
                roc_plot[j, i] = roc
                pr_plot[j, i] = pr

            save_attractor_results(i + 1)

    # endregion

    # region Weights (Save|Load)
    def save_weights(self):
        for model_name, model in self.saved_models.items():
            self._save_model_weights(model, model_name, self.epochs_seen)

    def _save_model_weights(self,
                            model: KerasModel,
                            name: str,
                            epoch: Union[int, None]):
        filepath = AutoEncoderBaseModel.get_model_weights_base_filepath(self.log_dir, name, epoch)
        model.save_weights(filepath)

    def load_weights(self,
                     base_filepath: str,
                     epoch: Union[int, None]):
        for model_name, model in self.saved_models.items():
            self._load_model_weights(model, base_filepath, model_name, epoch)

    @staticmethod
    def _load_model_weights(model: KerasModel,
                            base_filepath: str,
                            name: str,
                            epoch: int):
        filepath = AutoEncoderBaseModel.get_model_weights_base_filepath(base_filepath, name, epoch)
        model.load_weights(filepath)

    @staticmethod
    def get_model_weights_base_filepath(base_filepath: str, name: str, epoch: int = None):
        if epoch is None:
            return "{}/weights_{}.h5".format(base_filepath, name)
        else:
            return "{}/weights_{}_epoch_{}.h5".format(base_filepath, name, epoch)

    # endregion

    # region Callbacks
    # region Build callbacks
    def build_callbacks(self, dataset: DatasetLoader):
        return self.build_common_callbacks() + self.build_anomaly_callbacks(dataset)

    def build_common_callbacks(self) -> List[Callback]:
        assert self.tensorboard is None
        self.tensorboard = TensorBoard(log_dir=self.log_dir, update_freq=200, profile_batch=0)

        base_logger = BaseLogger()
        progbar_logger = ProgbarLogger(count_mode="steps")
        base_filepath = self.log_dir + "/weights_{model_name}_epoch_{epoch}.h5"
        model_checkpoint = MultipleModelsCheckpoint(base_filepath, self.saved_models, period=1)

        common_callbacks = [base_logger, self.tensorboard, progbar_logger, model_checkpoint]

        if "lr_schedule" in self.config:
            lr_scheduler = LearningRateScheduler(self.get_learning_rate_schedule(), verbose=1)
            common_callbacks.append(lr_scheduler)

        return common_callbacks

    def build_anomaly_callbacks(self, dataset: DatasetLoader) -> List[Callback]:
        # region Getting dataset/datasets
        test_subset = dataset.test_subset
        train_subset = dataset.train_subset
        # endregion

        train_image_callbacks = ImageCallback.make_video_autoencoder_callbacks(self.autoencoder,
                                                                               train_subset,
                                                                               name="train",
                                                                               is_train_callback=True,
                                                                               tensorboard=self.tensorboard,
                                                                               epoch_freq=1)
        test_image_callbacks = ImageCallback.make_video_autoencoder_callbacks(self.autoencoder,
                                                                              train_subset,
                                                                              name="test",
                                                                              is_train_callback=False,
                                                                              tensorboard=self.tensorboard,
                                                                              epoch_freq=1)

        anomaly_callbacks: List[Callback] = train_image_callbacks + test_image_callbacks

        # region AUC callback
        # TODO : Parameter for batch_size here
        from callbacks import AUCCallback
        inputs, outputs, labels = test_subset.get_batch(batch_size=1024, output_labels=True)

        labels = SubsetLoader.timestamps_labels_to_frame_labels(labels, inputs.shape[1])

        raw_predictions_model = self.get_anomalies_raw_predictions_model()
        frame_auc_callback = AUCCallback(raw_predictions_model, self.tensorboard,
                                         inputs, outputs, labels,
                                         plot_size=(128, 128), batch_size=16,
                                         name="Frame_Level_Error_AUC", epoch_freq=1)
        anomaly_callbacks.append(frame_auc_callback)
        # endregion

        return anomaly_callbacks

    def get_learning_rate_schedule(self):
        lr_drop_epochs = self.config["lr_schedule"]["drop_epochs"]
        lr_drop_factor = self.config["lr_schedule"]["drop_factor"]

        def schedule(epoch, learning_rate):
            if (epoch % lr_drop_epochs) == (lr_drop_epochs - 1):
                return learning_rate * (1.0 - lr_drop_factor)
            else:
                return learning_rate

        return schedule

    # endregion

    # region Setup callbacks
    def setup_callbacks(self,
                        callbacks: CallbackList or List[Callback],
                        batch_size: int,
                        epochs: int,
                        epoch_length: int) -> CallbackList:
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self.autoencoder)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': epoch_length,
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

    # endregion
    # endregion

    # region Log dir

    @classmethod
    def make_log_dir(cls,
                     dataset_name: str,
                     ) -> str:
        project_log_dir = "../logs/AutoEncoding-Anomalies"
        base_dir = os.path.join(project_log_dir, dataset_name, cls.__name__)
        log_dir = get_log_dir(base_dir)
        return log_dir

    @property
    def models_with_saved_info(self) -> List[KerasModel]:
        return [self.encoder, self.decoder, self.autoencoder]

    def save_models_info(self, log_dir: str):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        for model in self.models_with_saved_info:
            AutoEncoderBaseModel.save_model_info(log_dir, model)

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
    def channels_axis(self) -> int:
        return 0 if self.channels_first else -1

    @property
    def time_axis(self) -> int:
        return 2 if self.channels_first else 1

    @property
    def data_format(self) -> str:
        return "channels_first" if self.channels_first else "channels_last"

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self.input_shape[2:] if self.channels_first else self.input_shape[1:-1]

    @property
    def input_sequence_length(self) -> int:
        return self.input_shape[self.time_axis - 1]

    @property
    def output_image_size(self) -> Tuple[int, int]:
        return self.output_shape[2:] if self.channels_first else self.output_shape[1:-1]

    @property
    def output_sequence_length(self) -> int:
        return self.output_shape[1] if self.channels_first else self.output_shape[0]

    @property
    def seed(self) -> int:
        return self.config["seed"]
    # endregion
