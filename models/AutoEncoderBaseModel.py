from keras.models import Model as KerasModel
from keras.layers import Activation, LeakyReLU, Dense, Dropout
from keras.layers import Conv2D, Deconv2D, Conv3D, Deconv3D
from keras.regularizers import l1
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, CallbackList, Callback, ProgbarLogger, BaseLogger, LearningRateScheduler
from keras.backend import binary_crossentropy
from keras.utils import conv_utils
from keras.utils.generic_utils import to_list
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
import json
import copy
from typing import List, Any

from layers import ResBlock2D, ResBlock3D, ResBlock2DTranspose, ResBlock3DTranspose, SpectralNormalization
from datasets import Database, Dataset
from utils.train_utils import get_log_dir
from utils.summary_utils import image_summary
from callbacks import ImageCallback, AUCCallback, CallbackModel


class AutoEncoderScale(object):
    def __init__(self,
                 encoder: KerasModel,
                 decoder: KerasModel,
                 autoencoder: KerasModel,
                 **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder


uint8_max_constant = tf.constant(255.0, dtype=tf.float32)


class AutoEncoderBaseModel(ABC):
    # region Initialization
    def __init__(self):
        self.image_summaries_max_outputs = 3

        self.layers_built = False
        self.embeddings_layer = None
        self.encoder_layers = []
        self.decoder_layers = []

        self.config: dict = None

        self.input_shape = None
        self.output_shape = None
        self.channels_count = None

        self.embeddings_size = None
        self.use_dense_embeddings = None
        self.embeddings_shape = None
        self.embeddings_filters = None

        self.depth = 0
        self._scales: List[AutoEncoderScale] = []
        self._scales_input_shapes = None
        self._scales_output_shapes = None
        self._true_outputs_placeholders = None

        self.default_activation = None
        self.embeddings_activation = None
        self.output_activation = None

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

        embeddings_reshape = self.config["embeddings_reshape"]
        embeddings_reshape_dim = np.prod(embeddings_reshape)

        assert (self.embeddings_size % embeddings_reshape_dim) == 0, \
            "Embeddings size must be a multiple of {0} " \
            "(total dimension of embeddings reshape)".format(embeddings_reshape_dim)

        self.embeddings_filters = self.embeddings_size // embeddings_reshape_dim
        self.embeddings_shape = embeddings_reshape + [self.embeddings_filters]
        # endregion

        # region Scales
        self.depth = len(self.config["encoder"])
        self._scales: List[AutoEncoderScale] = [None] * self.depth
        self._true_outputs_placeholders = [None] * self.depth
        # endregion

        # region I/O shapes
        self.input_shape = self.config["input_shape"]
        assert 3 <= len(self.input_shape) <= 4

        self.channels_count = self.input_shape[-1]

        self.output_shape = self.output_shape_by_scale[-1]
        print(self.output_shape)
        assert 3 <= len(self.output_shape) <= 4
        # endregion

        # region Activations
        self.default_activation = self.config["default_activation"]
        self.embeddings_activation = self.config["embeddings_activation"]
        self.output_activation = self.config["output_activation"]
        # endregion

        # region Regularizers
        self.weight_decay_regularizer = l1(self.config["weight_decay"]) if "weight_decay" in self.config else None
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
        for layer_info in self.config["encoder"]:
            layer = self.build_conv_layer(layer_info, self.encoder_rank)
            self.encoder_layers.append(layer)

        if self.use_dense_embeddings:
            self.embeddings_layer = Dense(units=self.embeddings_size,
                                          kernel_regularizer=self.weight_decay_regularizer,
                                          bias_regularizer=self.weight_decay_regularizer)
        else:
            conv = conv_nd[False][False][self.encoder_rank]
            self.embeddings_layer = conv(filters=self.embeddings_filters, kernel_size=3, padding="same",
                                         kernel_regularizer=self.weight_decay_regularizer,
                                         bias_regularizer=self.weight_decay_regularizer)

        for layer_info in self.config["decoder"]:
            layer = self.build_deconv_layer(layer_info, self.decoder_rank)
            self.decoder_layers.append(layer)
        self.layers_built = True

    def build_conv_layer(self, layer_info: dict, rank: int, transpose=False):
        layer_kwargs = {"filters": layer_info["filters"], "kernel_size": layer_info["kernel_size"],
                        "strides": layer_info["strides"], "kernel_regularizer": self.weight_decay_regularizer,
                        "bias_regularizer": self.weight_decay_regularizer}
        use_resblock = ("resblock" in layer_info) and (layer_info["resblock"] == "True")
        if not use_resblock:
            layer_kwargs["padding"] = layer_info["padding"]

        layer = conv_nd[use_resblock][transpose][rank](**layer_kwargs)

        if self.use_spectral_norm:
            layer = SpectralNormalization(layer)
        return layer

    def build_deconv_layer(self, layer_info: dict, rank: int, ):
        return self.build_conv_layer(layer_info, rank, transpose=True)

    # endregion

    @abstractmethod
    def build_for_scale(self, scale: int):
        raise NotImplementedError

    def link_conv_layer(self, input_layer, layers_by_scale, layer_index, use_dropout, sub_config: dict):
        # Layer
        layer = layers_by_scale[layer_index](input_layer)

        # Activation
        layer = AutoEncoderBaseModel.get_activation(self.default_activation)(layer)

        # Dropout
        if use_dropout:
            if "dropout" in sub_config[layer_index]:
                layer = Dropout(rate=sub_config[layer_index]["dropout"])(layer)

        return layer

    # region Encoder models
    @abstractmethod
    def build_encoder_for_scale(self, scale: int):
        raise NotImplementedError

    def link_encoder_conv_layer(self, layer, scale: int, layer_index: int):
        use_dropout = layer_index > 0
        layer_index = layer_index + self.depth - scale - 1
        sub_config = self.config["encoder"]
        return self.link_conv_layer(layer, self.encoder_layers, layer_index, use_dropout, sub_config)

    # endregion

    # region Decoder models
    @abstractmethod
    def build_decoder_for_scale(self, scale: int):
        raise NotImplementedError

    def link_decoder_deconv_layer(self, layer, scale: int, layer_index: int):
        use_dropout = layer_index < scale
        sub_config = self.config["decoder"]
        return self.link_conv_layer(layer, self.decoder_layers, layer_index, use_dropout, sub_config)

    # endregion

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

    def get_scale(self, scale: int) -> AutoEncoderScale:
        if self._scales[scale] is None:
            self.build_for_scale(scale)
        return self._scales[scale]

    def get_autoencoder_model_at_scale(self, scale: int) -> KerasModel:
        return self.get_scale(scale).autoencoder

    def get_encoder_model_at_scale(self, scale: int) -> KerasModel:
        return self.get_scale(scale).encoder

    def get_decoder_model_at_scale(self, scale: int) -> KerasModel:
        return self.get_scale(scale).decoder

    def get_true_outputs_placeholder(self, scale: int) -> tf.Tensor:
        if self._true_outputs_placeholders[scale] is None:
            pred_output = self.get_autoencoder_model_at_scale(scale).output
            output_shape = self.output_shape_by_scale[scale]
            output_shape[-1] = self.channels_count

            true_outputs_placeholder = tf.placeholder(dtype=pred_output.dtype, shape=[None, *output_shape],
                                                      name="true_outputs_placeholder_scale_{0}".format(scale))
            self._true_outputs_placeholders[scale] = true_outputs_placeholder

        return self._true_outputs_placeholders[scale]

    # endregion

    # region Training
    def train(self,
              database: Database,
              batch_size: int or List[int] = 64,
              epoch_length: int = None,
              epochs: int or List[int] = 1,
              **kwargs):
        assert isinstance(batch_size, int) or isinstance(batch_size, list)
        assert isinstance(epochs, int) or isinstance(epochs, list)

        if not self.layers_built:
            self.build_layers()

        if self.log_dir is not None:
            return
        self.log_dir = self.__class__.make_log_dir(database)

        min_scale = kwargs.pop("min_scale") if "min_scale" in kwargs else 0
        max_scale = kwargs.pop("max_scale") if "max_scale" in kwargs else self.depth - 1

        model = self.get_autoencoder_model_at_scale(max_scale)
        self.save_model_info(self.log_dir, model)

        if isinstance(batch_size, int):
            batch_size = [batch_size] * (max_scale + 1)

        if isinstance(epochs, int):
            epochs = [epochs] * (max_scale + 1)

        samples_count = database.train_dataset.samples_count
        if epoch_length is None:
            epoch_length = [samples_count // scale_batch_size for scale_batch_size in batch_size]

        # region Pre-train
        common_callbacks = self.build_common_callbacks()
        common_callbacks = self.setup_callbacks(common_callbacks, model, batch_size[0], epochs[0],
                                                epoch_length, samples_count)

        common_callbacks.on_train_begin()
        if ("pre_train" not in kwargs) or kwargs["pre_train"]:
            self.pre_train_loop(database, common_callbacks, batch_size, epoch_length, epochs, min_scale, max_scale)
        # endregion

        # region Max scale training
        AutoEncoderBaseModel.update_callbacks_param(common_callbacks, "epochs", epochs[max_scale] + self.epochs_seen)
        anomaly_callbacks = self.build_anomaly_callbacks(database, scale=max_scale)
        anomaly_callbacks = self.setup_callbacks(anomaly_callbacks, model, batch_size[max_scale], epochs[max_scale],
                                                 epoch_length, samples_count)

        callbacks = CallbackList(common_callbacks.callbacks + anomaly_callbacks.callbacks)
        self.print_training_model_at_scale_header(max_scale, max_scale)

        anomaly_callbacks.on_train_begin()
        self.train_loop(database, callbacks, batch_size[max_scale], epoch_length,
                        epochs[max_scale], max_scale)
        callbacks.on_train_end()
        # endregion

    def resize_database(self, database: Database,
                        scale: int) -> Database:
        input_image_size = self.input_image_size_by_scale(scale)
        input_sequence_length = self.input_sequence_length_by_scale(scale)
        output_sequence_length = self.output_sequence_length_by_scale(scale)
        database = database.resized_to_scale(input_image_size, input_sequence_length, output_sequence_length)
        return database

    def pre_train_loop(self,
                       database: Database,
                       callbacks: CallbackList,
                       batch_size: List[int],
                       epoch_length,
                       epochs: List[int],
                       min_scale: int,
                       max_scale: int):
        for scale in range(min_scale, max_scale):
            AutoEncoderBaseModel.update_callbacks_param(callbacks, "epochs", epochs[scale] + self.epochs_seen)
            AutoEncoderBaseModel.update_callbacks_param(callbacks, "batch_size", batch_size[scale])
            self.print_training_model_at_scale_header(scale, max_scale)
            self.pre_train_scale(database, callbacks, scale, batch_size[scale], epoch_length, epochs[scale])

    def pre_train_scale(self,
                        database: Database,
                        callbacks: CallbackList,
                        scale: int,
                        batch_size: int,
                        epoch_length: int,
                        epochs: int):
        self.train_loop(database, callbacks, batch_size, epoch_length, epochs, scale)

    def train_loop(self,
                   database: Database,
                   callbacks: CallbackList,
                   batch_size: int,
                   epoch_length: int,
                   epochs: int,
                   scale: int):
        database = self.resize_database(database, scale)
        database.train_dataset.epoch_length = epoch_length
        database.train_dataset.batch_size = batch_size
        database.test_dataset.batch_size = batch_size

        for _ in range(epochs):
            self.train_epoch(database, scale, callbacks)

    def train_epoch(self,
                    database: Database,
                    scale: int = None,
                    callbacks: CallbackList = None):
        epoch_length = len(database.train_dataset)
        model = self.get_autoencoder_model_at_scale(scale)

        callbacks.on_epoch_begin(self.epochs_seen)

        for batch_index in range(epoch_length):
            x, y = database.train_dataset[0]

            batch_logs = {"batch": batch_index, "size": x.shape[0]}
            callbacks.on_batch_begin(batch_index, batch_logs)

            results = model.train_on_batch(x=x, y=y)

            if "metrics" in self.config:
                batch_logs["loss"] = results[0]
                for metric_name, result in zip(self.config["metrics"], results[1:]):
                    batch_logs[metric_name] = result
            else:
                batch_logs["loss"] = results

            callbacks.on_batch_end(batch_index, batch_logs)

        self.on_epoch_end(model, database, callbacks)

    def on_epoch_end(self,
                     base_model: KerasModel,
                     database: Database,
                     callbacks: CallbackList = None,
                     epoch_logs: dict = None):
        if epoch_logs is None:
            epoch_logs = {}

        out_labels = base_model.metrics_names
        val_outs = base_model.evaluate_generator(database.test_dataset)
        val_outs = to_list(val_outs)
        for label, val_out in zip(out_labels, val_outs):
            epoch_logs["val_{0}".format(label)] = val_out

        database.on_epoch_end()

        if callbacks:
            callbacks.on_epoch_end(self.epochs_seen, epoch_logs)
        self.epochs_seen += 1

    def print_training_model_at_scale_header(self,
                                             scale: int,
                                             max_scale: int):
        scale_shape = self.input_image_size_by_scale(scale)
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

        base_logger = BaseLogger()
        progbar_logger = ProgbarLogger(count_mode="steps")

        common_callbacks = [base_logger, self.tensorboard, progbar_logger]

        if ("lr_drop_epochs" in self.config) and (self.config["lr_drop_epochs"] > 0):
            lr_scheduler = LearningRateScheduler(self.get_learning_rate_schedule())
            common_callbacks.append(lr_scheduler)

        return common_callbacks

    def build_anomaly_callbacks(self,
                                database: Database,
                                scale: int = None):
        database = self.resize_database(database, scale)
        test_dataset = database.test_dataset

        train_image_callback = self.image_callback_from_dataset(database.train_dataset, "train",
                                                                self.tensorboard, scale=scale)
        eval_image_callback = self.image_callback_from_dataset(test_dataset, "test",
                                                               self.tensorboard, scale=scale)

        samples = test_dataset.sample(batch_size=512, seed=16, max_shard_count=8, return_labels=True)
        auc_images, frame_labels, pixel_labels = samples
        auc_images = test_dataset.divide_batch_io(auc_images)

        frame_predictions_model = self.build_frame_level_error_callback_model(scale)
        frame_auc_callback = AUCCallback(frame_predictions_model, self.tensorboard,
                                         auc_images, frame_labels,
                                         plot_size=(256, 256), batch_size=128,
                                         name="Frame_Level_Error_AUC")

        anomaly_callbacks = [train_image_callback, eval_image_callback, frame_auc_callback]

        if pixel_labels is not None:
            pixel_predictions_model = self.build_pixel_level_error_callback_model(scale)
            pixel_auc_callback = AUCCallback(pixel_predictions_model, self.tensorboard,
                                             auc_images, pixel_labels,
                                             plot_size=(256, 256), batch_size=128,
                                             num_thresholds=20, name="Pixel_Level_Error_AUC")
            anomaly_callbacks.append(pixel_auc_callback)

        return anomaly_callbacks

    def setup_callbacks(self,
                        callbacks: CallbackList or List[Callback],
                        model: KerasModel,
                        batch_size: int,
                        epochs: int,
                        epoch_length: int,
                        samples_count: int) -> CallbackList:
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

    def callback_metrics(self, model: KerasModel):
        metrics_names = model.metrics_names
        validation_callbacks = ["val_{0}".format(name) for name in metrics_names]
        callback_metrics = copy.copy(metrics_names) + validation_callbacks
        return callback_metrics

    def image_callback_from_dataset(self,
                                    dataset: Dataset,
                                    name: str,
                                    tensorboard: TensorBoard,
                                    frequency="epoch",
                                    scale: int = None) -> ImageCallback:

        images = dataset.get_batch(self.image_summaries_max_outputs, seed=0, apply_preprocess_step=False,
                                   max_shard_count=self.image_summaries_max_outputs)

        autoencoder = self.get_autoencoder_model_at_scale(scale)
        pred_outputs = autoencoder.output
        true_outputs = self.get_true_outputs_placeholder(scale)

        summary_op = self.make_images_summary(name, autoencoder.input, true_outputs, pred_outputs)

        summary_model_inputs = [autoencoder.input, true_outputs]
        summary_model = CallbackModel(inputs=summary_model_inputs, outputs=summary_op, output_is_summary=True)

        return ImageCallback(summary_model, images, tensorboard, frequency)

    def make_images_summary(self,
                            name: str,
                            inputs: tf.Tensor,
                            true_outputs: tf.Tensor,
                            pred_outputs: tf.Tensor):
        inputs = self.normalize_image_tensor(inputs)
        true_outputs = self.normalize_image_tensor(true_outputs)
        pred_outputs = self.normalize_image_tensor(pred_outputs)
        io_delta = (pred_outputs - true_outputs) * (tf.cast(pred_outputs < true_outputs, dtype=tf.uint8) * 254 + 1)

        max_outputs = self.image_summaries_max_outputs
        summaries = [image_summary(name + "_inputs", inputs, max_outputs, fps=5),
                     image_summary(name + "_true_outputs", true_outputs, max_outputs, fps=5),
                     image_summary(name + "_pred_outputs", true_outputs, max_outputs, fps=5),
                     image_summary(name + "_delta", io_delta, max_outputs, fps=5)]
        return tf.summary.merge(summaries)

    def normalize_image_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("normalize_image_tensor"):
            normalized = (tensor - self._min_output_constant) * self._inv_output_range_constant
            normalized = tf.cast(normalized * uint8_max_constant, tf.uint8)
        return normalized

    def build_pixel_level_error_callback_model(self, scale: int = None):
        model = self.get_autoencoder_model_at_scale(scale)
        true_outputs = self.get_true_outputs_placeholder(scale)

        error = tf.abs(model.output - true_outputs)

        pixel_predictions_model = CallbackModel([model.input, true_outputs], error)

        return pixel_predictions_model

    def build_frame_level_error_callback_model(self, scale: int = None):
        model = self.get_autoencoder_model_at_scale(scale)
        true_outputs = self.get_true_outputs_placeholder(scale)

        squared_delta = tf.abs(model.output - true_outputs)
        average_error = tf.reduce_max(squared_delta, axis=[-3, -2, -1])

        frame_predictions_model = CallbackModel([model.input, true_outputs], average_error)

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

    # region I/O Shape by scale
    @property
    def input_shape_by_scale(self):
        if self._scales_input_shapes is None:
            input_shape = self.input_shape
            self._scales_input_shapes = []

            for layer_info in self.config["encoder"]:
                self._scales_input_shapes.append(input_shape)

                space = input_shape[:-1]
                kernel_size = conv_utils.normalize_tuple(layer_info["kernel_size"], self.encoder_rank, "kernel_size")
                strides = conv_utils.normalize_tuple(layer_info["strides"], self.encoder_rank, "strides")

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

    def input_image_size_by_scale(self, scale):
        shape = self.input_shape_by_scale[scale]
        return shape[self.encoder_rank - 2:-1]

    def input_sequence_length_by_scale(self, scale):
        if self.encoder_rank == 2:
            return None
        return self.input_shape_by_scale[scale][0]

    @property
    def output_shape_by_scale(self):
        if self._scales_output_shapes is None:
            output_shape = self.embeddings_shape
            self._scales_output_shapes = []

            for j in range(self.depth):
                layer_info = self.config["decoder"][j]
                space = output_shape[:-1]
                kernel_size = conv_utils.normalize_tuple(layer_info["kernel_size"], self.decoder_rank, "kernel_size")
                strides = conv_utils.normalize_tuple(layer_info["strides"], self.decoder_rank, "strides")

                new_space = []
                for i in range(len(space)):
                    dim = conv_utils.deconv_length(space[i],
                                                   strides[i],
                                                   kernel_size[i],
                                                   padding=layer_info["padding"],
                                                   output_padding=None)
                    new_space.append(dim)
                filters = self.config["decoder"][j + 1]["filters"] if j < (self.depth - 1) else self.channels_count
                output_shape = [*new_space, filters]
                self._scales_output_shapes.append(output_shape)

        return self._scales_output_shapes

    def output_image_size_by_scale(self, scale):
        shape = self.output_shape_by_scale[scale]
        return shape[self.decoder_rank - 2:-1]

    def output_sequence_length_by_scale(self, scale):
        if self.decoder_rank == 2:
            return None
        return self.output_shape_by_scale[scale][0]

    # endregion

    # region Log dir

    @classmethod
    def make_log_dir(cls,
                     database: Database):
        project_log_dir = "../logs/AutoEncoding-Anomalies"
        base_dir = os.path.join(project_log_dir, database.__class__.__name__, cls.__name__)
        log_dir = get_log_dir(base_dir)
        return log_dir

    def save_model_info(self,
                        log_dir: str,
                        model: KerasModel):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

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

    # region Properties
    @property
    def image_size(self):
        if self.input_is_sequence:
            return self.input_shape[1:3]
        else:
            return self.input_shape[:2]

    @property
    def encoder_rank(self):
        return len(self.input_shape) - 1

    @property
    def decoder_rank(self):
        return len(self.embeddings_shape) - 1

    @property
    def input_is_sequence(self):
        return self.encoder_rank > 2

    @property
    def output_is_sequence(self):
        return self.decoder_rank > 2
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
conv_nd = {False: {False: {2: Conv2D, 3: Conv3D},
                   True: {2: Deconv2D, 3: Deconv3D}},
           True: {False: {2: ResBlock2D, 3: ResBlock3D},
                  True: {2: ResBlock2DTranspose, 3: ResBlock3DTranspose}}
           }


def build_adaptor_layer(channels: int, rank: int):
    layer_class = conv_nd[False][False][rank]
    return layer_class(filters=channels, kernel_size=1, strides=1, padding="same")
# endregion
