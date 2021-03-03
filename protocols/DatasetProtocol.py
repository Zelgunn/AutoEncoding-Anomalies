import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import Model, optimizers
import numpy as np
from abc import abstractmethod
from typing import Dict, List, Union, Callable
import json
import os
from shutil import copyfile

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Pattern
from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols.utils import make_encoder, make_decoder, make_discriminator
from callbacks.configs import AUCCallbackConfig, AnomalyDetectorCallbackConfig
from custom_tf_models import AE, IAE, LED
from custom_tf_models.energy_based import EBAE
from custom_tf_models.adversarial import IAEGAN


class DatasetProtocol(Protocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 base_log_dir: str,
                 epoch: int,
                 ):
        self.config = self.load_config(protocol_name, dataset_name)
        if "seed" not in self.config:
            self.config["seed"] = int(np.random.randint(low=0, high=2 ** 31, dtype=np.int32))
        output_range = (-1.0, 1.0) if self.output_activation == "tanh" else (0.0, 1.0)
        self.epoch = epoch

        super(DatasetProtocol, self).__init__(dataset_name=dataset_name,
                                              protocol_name=protocol_name,
                                              base_log_dir=base_log_dir,
                                              model=None,
                                              output_range=output_range,
                                              seed=self.config["seed"])

    @property
    def model_name(self) -> str:
        return self.model_architecture

    # region Train
    def train_model(self, config: ProtocolTrainConfig = None, **kwargs):
        if config is None:
            config = self.get_train_config()

        super(DatasetProtocol, self).train_model(config)

    def get_train_config(self) -> ProtocolTrainConfig:
        train_pattern = self.get_train_pattern()
        modality_callback_configs = self.get_modality_callback_configs()
        auc_callbacks_configs = self.get_auc_callback_configs()
        anomaly_detector_callback_configs = self.get_anomaly_detector_callback_configs()

        return ProtocolTrainConfig(batch_size=self.batch_size,
                                   pattern=train_pattern,
                                   steps_per_epoch=self.steps_per_epoch,
                                   epochs=self.epochs,
                                   initial_epoch=self.epoch,
                                   validation_steps=self.validation_steps,
                                   save_frequency=self.save_frequency,
                                   modality_callback_configs=modality_callback_configs,
                                   auc_callback_configs=auc_callbacks_configs,
                                   anomaly_detector_callback_configs=anomaly_detector_callback_configs)

    def get_modality_callback_configs(self):
        return None

    # endregion

    # region Test
    def test_model(self, config: ProtocolTestConfig = None):
        if config is None:
            config = self.get_test_config()

        super(DatasetProtocol, self).test_model(config)

    @abstractmethod
    def get_test_config(self) -> ProtocolTestConfig:
        raise NotImplementedError

    # endregion

    # region Patterns
    @abstractmethod
    def get_train_pattern(self) -> Pattern:
        raise NotImplementedError

    @abstractmethod
    def get_anomaly_pattern(self) -> Pattern:
        raise NotImplementedError

    # endregion

    # region Callbacks
    def get_auc_callback_configs(self) -> List[AUCCallbackConfig]:
        if (self.auc_frequency < 1) or isinstance(self.model, LED):
            return []

        anomaly_pattern = self.get_anomaly_pattern()
        auc_callbacks_configs = []

        model = self.model
        if isinstance(model, EBAE):
            model = model.autoencoder

        elif isinstance(model, AE):
            auc_callbacks_configs += [
                AUCCallbackConfig(model, anomaly_pattern, labels_length=self.output_length, prefix="AE",
                                  convert_to_io_compare_model=True, epoch_freq=self.auc_frequency,
                                  io_compare_metrics="clipped_mae", sample_count=self.auc_sample_count),
            ]

        if isinstance(model, IAE):
            auc_callbacks_configs += [
                AUCCallbackConfig(model.interpolate, anomaly_pattern, labels_length=self.output_length, prefix="IAE",
                                  convert_to_io_compare_model=True, epoch_freq=self.auc_frequency,
                                  io_compare_metrics="clipped_mae", sample_count=self.auc_sample_count)
            ]

        if isinstance(model, IAEGAN):
            auc_callbacks_configs += [
                AUCCallbackConfig(model.discriminate, anomaly_pattern, labels_length=1, prefix="GAN",
                                  convert_to_io_compare_model=False, epoch_freq=self.auc_frequency,
                                  sample_count=self.auc_sample_count)
            ]

        return auc_callbacks_configs

    def get_anomaly_detector_callback_configs(self) -> List[AnomalyDetectorCallbackConfig]:
        callbacks = []
        if isinstance(self.model, LED):
            callback = AnomalyDetectorCallbackConfig(autoencoder=self.model,
                                                     pattern=self.get_anomaly_pattern(),
                                                     compare_metrics=None,
                                                     additional_metrics=self.model.additional_test_metrics,
                                                     stride=1,
                                                     epoch_freq=self.auc_frequency,
                                                     pre_normalize_predictions=True,
                                                     max_samples=-1,
                                                     )
            callbacks.append(callback)
        return callbacks

    # endregion

    def make_log_dir(self, sub_folder: str) -> str:
        log_dir = super(DatasetProtocol, self).make_log_dir(sub_folder)
        self.save_model_config(log_dir)
        self.save_dataset_config(log_dir)
        return log_dir

    # region Config
    @property
    @abstractmethod
    def output_length(self) -> int:
        raise NotImplementedError

    def get_config_path(self, protocol_name: str = None, dataset_name: str = None):
        if protocol_name is None:
            protocol_name = self.protocol_name
        if dataset_name is None:
            protocol_name = self.dataset_name
        return "protocols/configs/{protocol_name}/{dataset_name}.json" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

    def load_config(self, protocol_name: str = None, dataset_name: str = None) -> Dict:
        config_path = self.get_config_path(protocol_name, dataset_name)
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config

    def save_model_config(self, log_dir: str):
        config_path = os.path.join(log_dir, "main_config.json")
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file)

    def save_dataset_config(self, log_dir: str):
        source_path = os.path.join(self.dataset_folder, tfrecords_config_filename)
        target_path = os.path.join(log_dir, "dataset_{}".format(tfrecords_config_filename))
        copyfile(src=source_path, dst=target_path)

    # region Training
    @property
    def batch_size(self) -> int:
        return self.config["batch_size"]

    @property
    def output_activation(self) -> str:
        return self.config["output_activation"]

    @property
    def epochs(self) -> int:
        return int(self.config["epochs"])

    @property
    def steps_per_epoch(self) -> int:
        return int(self.config["steps_per_epoch"])

    @property
    def validation_steps(self) -> int:
        return int(self.config["validation_steps"])

    @property
    def save_frequency(self) -> Union[str, int]:
        save_frequency = self.config["save_frequency"]
        if save_frequency not in ["batch", "epoch"]:
            save_frequency = int(save_frequency)
        return save_frequency

    @property
    def auc_frequency(self) -> int:
        if "auc_frequency" not in self.config:
            self.config["auc_frequency"] = 1
        return self.config["auc_frequency"]

    @property
    def auc_sample_count(self) -> int:
        if "auc_sample_count" not in self.config:
            self.config["auc_sample_count"] = 128
        return self.config["auc_sample_count"]

    # endregion

    @property
    def channels(self) -> int:
        return self.config["channels"]

    # region Encoder
    @property
    def encoder_config(self):
        return self.config["encoder"]

    @property
    def encoder_mode(self):
        return self.encoder_config["mode"]

    @property
    def encoder_filters(self) -> List[int]:
        return self.encoder_config["filters"]

    @property
    def encoder_strides(self) -> List[List[int]]:
        return self.encoder_config["strides"]

    @property
    def encoder_kernel_sizes(self) -> List[int]:
        layer_count = len(self.encoder_filters)
        kernel_sizes = [self.base_kernel_size] * layer_count
        if "stem_kernel_size" in self.config:
            kernel_sizes[0] = self.stem_kernel_size
        return kernel_sizes

    @property
    def code_size(self) -> int:
        return self.config["code_size"]

    @property
    def code_activation(self) -> str:
        return self.config["code_activation"]

    # endregion

    # region Decoder
    @property
    def decoder_config(self):
        return self.config["decoder"]

    @property
    def decoder_mode(self):
        return self.decoder_config["mode"]

    @property
    def decoder_filters(self) -> List[int]:
        return self.decoder_config["filters"]

    @property
    def decoder_strides(self) -> List[List[int]]:
        return self.decoder_config["strides"]

    @property
    def decoder_kernel_sizes(self) -> List[int]:
        layer_count = len(self.decoder_filters)
        kernel_sizes = [self.base_kernel_size] * layer_count
        return kernel_sizes

    # endregion

    # region Discriminator
    @property
    def discriminator_config(self):
        return self.config["discriminator"]

    @property
    def discriminator_mode(self):
        return self.discriminator_config["mode"]

    @property
    def discriminator_filters(self) -> List[int]:
        return self.discriminator_config["filters"]

    @property
    def discriminator_strides(self) -> List[List[int]]:
        return self.discriminator_config["strides"]

    # endregion

    @property
    def model_architecture(self) -> str:
        return self.config["model_architecture"].lower()

    @property
    def base_kernel_size(self) -> int:
        return self.config["base_kernel_size"]

    @property
    def stem_kernel_size(self) -> int:
        return self.config["stem_kernel_size"]

    @property
    def basic_block_count(self) -> int:
        return self.config["basic_block_count"]

    # region Optimizer / Learning rate
    @property
    def optimizer_class(self) -> str:
        return self.config["optimizer"].lower()

    @property
    def learning_rate(self) -> float:
        return self.config["learning_rate"]

    @property
    def base_learning_rate_schedule(self):
        # from misc_utils.train_utils import WarmupSchedule

        learning_rate = self.learning_rate
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 10000, 0.25, staircase=True)
        # learning_rate = WarmupSchedule(warmup_steps=1000, learning_rate=learning_rate)
        # min_learning_rate = ScaledSchedule(learning_rate, 1e-2)
        # learning_rate = CyclicSchedule(cycle_length=1000,
        #                                learning_rate=min_learning_rate,
        #                                max_learning_rate=learning_rate)
        return learning_rate

    @property
    def discriminator_learning_rate_schedule(self):
        learning_rate = self.base_learning_rate_schedule
        # learning_rate = ScaledSchedule(learning_rate, 1.0)
        return learning_rate

    # endregion
    # endregion

    # region Make sub-models
    # region Base
    def make_encoder(self, input_shape, name="Encoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               mode=self.encoder_mode,
                               filters=self.encoder_filters,
                               kernel_size=self.encoder_kernel_sizes,
                               strides=self.encoder_strides,
                               code_size=self.code_size,
                               code_activation=self.code_activation,
                               basic_block_count=self.basic_block_count,
                               name=name,
                               )
        return encoder

    def make_decoder(self, input_shape, name="Decoder") -> Model:
        decoder = make_decoder(input_shape=input_shape,
                               mode=self.decoder_mode,
                               filters=self.decoder_filters,
                               kernel_size=self.decoder_kernel_sizes,
                               stem_kernel_size=self.stem_kernel_size,
                               strides=self.decoder_strides,
                               channels=self.channels,
                               output_activation=self.output_activation,
                               basic_block_count=self.basic_block_count,
                               name=name,
                               )
        return decoder

    # endregion

    # region Adversarial
    def make_discriminator(self, input_shape) -> Model:
        include_intermediate_output = self.model_architecture in ["vaegan", "avp"]
        discriminator = make_discriminator(input_shape=input_shape,
                                           mode=self.discriminator_mode,
                                           filters=self.discriminator_filters,
                                           kernel_size=self.base_kernel_size,
                                           strides=self.discriminator_strides,
                                           intermediate_size=self.discriminator_config["intermediate_size"],
                                           intermediate_activation="relu",
                                           include_intermediate_output=include_intermediate_output,
                                           basic_block_count=self.basic_block_count)
        return discriminator

    # endregion
    # endregion

    # region Optimizers
    def make_optimizer(self,
                       learning_rate: Union[Callable, float],
                       optimizer_class: str = None
                       ) -> optimizers.optimizer_v2.OptimizerV2:

        optimizer_class = self.optimizer_class if optimizer_class is None else optimizer_class
        if optimizer_class == "adam":
            return tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_class == "adamw":
            return tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=learning_rate)
        elif optimizer_class == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate)
        elif optimizer_class == "sgd":
            return tf.keras.optimizers.SGD(learning_rate)
        else:
            raise ValueError("`{}` is not a valid optimizer identifier.".format(optimizer_class))

    def make_base_optimizer(self) -> optimizers.optimizer_v2.OptimizerV2:
        return self.make_optimizer(self.base_learning_rate_schedule)

    def make_discriminator_optimizer(self) -> optimizers.optimizer_v2.OptimizerV2:
        if "discriminator_optimizer" in self.config:
            discriminator_optimizer_class = self.config["discriminator_optimizer"]
        else:
            discriminator_optimizer_class = self.optimizer_class
        return self.make_optimizer(self.discriminator_learning_rate_schedule, discriminator_optimizer_class)

    # endregion
