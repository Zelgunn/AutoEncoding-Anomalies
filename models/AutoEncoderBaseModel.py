from keras.models import Model as KerasModel
from keras.layers import Activation, LeakyReLU, Dense, Dropout, Layer, Input
from keras.layers import Conv3D, Deconv3D, MaxPooling3D, AveragePooling3D, UpSampling3D
from keras.layers import Concatenate
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, CallbackList, Callback, ProgbarLogger, BaseLogger, LearningRateScheduler
from keras.backend import binary_crossentropy
from keras.utils.generic_utils import to_list
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
import json
import copy
from typing import List, Any

from layers import ResBlock3D, ResBlock3DTranspose, DenseBlock3D, SpectralNormalization
from datasets import Database, Dataset
from utils.train_utils import get_log_dir
from utils.summary_utils import image_summary
from utils.test_utils import visualize_model_errors, evaluate_model_anomaly_detection
from callbacks import ImageCallback, AUCCallback, CallbackModel


# region Containers
class LayerBlock(object):
    def __init__(self,
                 conv_layer: Layer,
                 pool_layer: Layer,
                 upsampling_layer: Layer,
                 activation_layer: Layer,
                 dropout_layer: Dropout,
                 is_transpose: bool):
        self.conv = conv_layer
        self.pooling = pool_layer
        self.upsampling_layer = upsampling_layer
        self.activation = activation_layer
        self.dropout = dropout_layer

        self.is_transpose = is_transpose

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

        self._true_outputs_placeholder = None

        self.default_activation = None
        self.embeddings_activation = None
        self.output_activation = None

        self.weights_initializer = None

        self.block_type_name = None
        self.pooling = None
        self.upsampling = None
        self.use_batch_normalization = None

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

    # endregion

    # region Model(s) (Builders)

    # region Layers
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
            else:
                conv_layer_kwargs["padding"] = stack_info["padding"] if "padding" in stack_info else "same"

            conv_block_type = self.get_block_type(transpose)
            conv_layer = conv_block_type(**conv_layer_kwargs)
            # endregion

            # region Spectral normalization
            if self.use_spectral_norm:
                conv_layer = SpectralNormalization(conv_layer)
            # endregion

            activation = AutoEncoderBaseModel.get_activation(self.default_activation)
            dropout = Dropout(rate=stack_info["dropout"]) if "dropout" in stack_info else None

            layer_block = LayerBlock(conv_layer, pool_layer, upsampling_layer, activation, dropout, transpose)
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

    # endregion

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def build_encoder(self):
        raise NotImplementedError

    def build_decoder_half(self, input_layer, predictor_half):
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

    def build_decoder(self):
        self.decoder_input_layer = Input(self.compute_decoder_input_shape())

        self.reconstructor_output = self.build_decoder_half(self.decoder_input_layer, False)
        self.predictor_output = self.build_decoder_half(self.decoder_input_layer, True)

        output_layer = Concatenate(axis=1)([self.reconstructor_output, self.predictor_output])

        self._decoder = KerasModel(inputs=self.decoder_input_layer, outputs=output_layer, name="Decoder")

    @staticmethod
    def get_activation(activation_config: dict):
        if activation_config["name"] == "leaky_relu":
            return LeakyReLU(alpha=activation_config["alpha"])
        else:
            return Activation(activation_config["name"])

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
            self.build()
        return self._autoencoder

    @property
    def encoder(self):
        if self._encoder is None:
            self.build_encoder()
        return self._encoder

    @property
    def decoder(self):
        if self._decoder is None:
            self.build_decoder()
        return self._decoder

    # endregion

    # region Training
    def train(self,
              database: Database,
              batch_size: int = 64,
              epoch_length: int = None,
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

        samples_count = database.train_dataset.samples_count
        if epoch_length is None:
            epoch_length = samples_count // batch_size

        callbacks = self.build_callbacks(database)
        callbacks = self.setup_callbacks(callbacks, batch_size, epochs, epoch_length, samples_count)

        callbacks.on_train_begin()
        try:
            self.train_loop(database, callbacks, batch_size, epoch_length, epochs)
        except KeyboardInterrupt:
            print("\n==== Training was stopped by a Keyboard Interrupt ====\n")
        callbacks.on_train_end()

        print("============== Saving models weights... ==============")
        self.save_weights("weights".format(self.epochs_seen))
        print("======================= Done ! =======================")

        visualize_model_errors(self.autoencoder, database.test_dataset)
        evaluate_model_anomaly_detection(self.autoencoder, database.test_dataset, epoch_length, batch_size, True)

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

    def save_weights(self, base_filename):
        self._save_model_weights(self.encoder, base_filename, "encoder")
        self._save_model_weights(self.decoder, base_filename, "decoder")

    def _save_model_weights(self, model: KerasModel, base_filename, name):
        filepath = "{}{sep}{}_{}.h5".format(self.log_dir, base_filename, name, sep=os.path.sep)
        model.save_weights(filepath)

    def load_weights(self, base_filepath):
        AutoEncoderBaseModel._load_model_weights(self.encoder, base_filepath, "encoder")
        AutoEncoderBaseModel._load_model_weights(self.decoder, base_filepath, "decoder")

    @staticmethod
    def _load_model_weights(model: KerasModel, base_filepath, name):
        filepath = "{}_{}.h5".format(base_filepath, name)
        model.load_weights(filepath)

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
        samples = test_dataset.sample(batch_size=512, seed=16, max_shard_count=8, return_labels=True)
        auc_images, frame_labels, pixel_labels = samples
        auc_images = test_dataset.divide_batch_io(auc_images)
        # endregion

        # region Frame level error AUC (ROC)
        frame_predictions_model = self.build_frame_level_error_callback_model()
        frame_auc_callback = AUCCallback(frame_predictions_model, self.tensorboard,
                                         auc_images, frame_labels,
                                         plot_size=(128, 128), batch_size=32,
                                         name="Frame_Level_Error_AUC", epoch_freq=5)
        # endregion

        anomaly_callbacks = [train_image_callback, test_image_callback, frame_auc_callback]

        # region Pixel level error AUC (ROC)
        # TODO : Check labels size in UCSDDatabase
        # if pixel_labels is not None:
        #     pixel_auc_callback = AUCCallback(pixel_predictions_model, self.tensorboard,
        #                                      auc_images, pixel_labels,
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

        images = dataset.get_batch(self.image_summaries_max_outputs, seed=0, apply_preprocess_step=False,
                                   max_shard_count=self.image_summaries_max_outputs)

        # region Placeholders
        true_outputs_placeholder = self.get_true_outputs_placeholder()
        summary_inputs = [self.autoencoder.input, true_outputs_placeholder]
        # endregion

        # region Image/Video normalization (uint8)
        inputs = self.normalize_image_tensor(self.autoencoder.input)
        true_outputs = self.normalize_image_tensor(true_outputs_placeholder)
        pred_outputs = self.normalize_image_tensor(self.autoencoder.output)
        # endregion

        io_delta = (pred_outputs - true_outputs) * (tf.cast(pred_outputs < true_outputs, dtype=tf.uint8) * 254 + 1)

        # region Summary operations
        max_outputs = self.image_summaries_max_outputs
        one_shot_summaries = [image_summary(name + "_inputs", inputs, max_outputs, fps=5),
                              image_summary(name + "_true_outputs", true_outputs, max_outputs, fps=5)]
        repeated_summaries = [image_summary(name + "_pred_outputs", pred_outputs, max_outputs, fps=5),
                              image_summary(name + "_delta", io_delta, max_outputs, fps=5)]
        # endregion

        one_shot_summary_model = CallbackModel(summary_inputs, one_shot_summaries, output_is_summary=True)
        repeated_summary_model = CallbackModel(summary_inputs, repeated_summaries, output_is_summary=True)

        return ImageCallback(one_shot_summary_model, repeated_summary_model, images, tensorboard, frequency)

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
        average_error = tf.reduce_max(squared_delta, axis=[-3, -2, -1])

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

# region Dynamic choice between Conv2D/Deconv2D and Conv3D/Deconv3D
types_with_transpose = ["conv_block", "residual_block"]

conv_type = {"conv_block": {False: Conv3D, True: Deconv3D},
             "residual_block": {False: ResBlock3D, True: ResBlock3DTranspose},
             "dense_block": {False: DenseBlock3D}}

pool_type = {"max": MaxPooling3D,
             "average": AveragePooling3D}
# endregion
