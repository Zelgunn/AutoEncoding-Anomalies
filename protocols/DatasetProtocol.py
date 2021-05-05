import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import Model, optimizers
import numpy as np
from abc import abstractmethod
from typing import Dict, List, Union, Callable, Tuple, Any
import json
import os
from shutil import copyfile

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Pattern
from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols.utils import make_encoder, make_decoder, make_discriminator
from callbacks.configs import AUCCallbackConfig, AnomalyDetectorCallbackConfig
from custom_tf_models import AE, VAE, IAE, VIAE, LED, CnC, IterativeAE
from custom_tf_models.energy_based import EBAE
from custom_tf_models.adversarial import IAEGAN


class DatasetProtocol(Protocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 base_log_dir: str,
                 epoch: int,
                 config: Dict = None,
                 ):
        if config is None:
            config = self.load_config(protocol_name, dataset_name)
        self.config = config
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

        if isinstance(model, CnC):
            auc_callbacks_configs += [
                AUCCallbackConfig(model.mean_relevance_energy, anomaly_pattern, labels_length=1, prefix="CnC",
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
    def get_config_path(self, protocol_name: str = None, dataset_name: str = None):
        if protocol_name is None:
            protocol_name = self.protocol_name
        if dataset_name is None:
            protocol_name = self.dataset_name
        return "protocols/configs/{protocol_name}/{dataset_name}.json" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

    def load_config(self, protocol_name: str = None, dataset_name: str = None) -> Dict:
        config_path = self.get_config_path(protocol_name, dataset_name)
        try:
            with open(config_path) as config_file:
                config = json.load(config_file)
        except json.decoder.JSONDecodeError:
            raise RuntimeError("Could not load protocol config from {}".format(config_path))
        return config

    def save_model_config(self, log_dir: str):
        config_path = os.path.join(log_dir, "main_config.json")
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file)

    def save_dataset_config(self, log_dir: str):
        source_path = os.path.join(self.dataset_folder, tfrecords_config_filename)
        target_path = os.path.join(log_dir, "dataset_{}".format(tfrecords_config_filename))
        copyfile(src=source_path, dst=target_path)

    def get_config_value(self, key: str, default: Any) -> Any:
        if key not in self.config:
            self.config[key] = default
        return self.config[key]

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
        return self.get_config_value("auc_frequency", default=1)

    @property
    def auc_sample_count(self) -> int:
        return self.get_config_value("auc_sample_count", default=128)

    # endregion

    # region Shapes
    @property
    def encoder_input_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError("This currently must be defined in sub-classes.")

    # region Time (step_size, step_count, ...)
    @property
    def input_length(self) -> int:
        if self.model_architecture in ["aep", "preled"]:
            return self.step_size
        else:
            return self.output_length

    @property
    def output_length(self) -> int:
        if self.model_architecture in ["iae", "viae", "and", "iaegan", "avp"]:
            return self.step_size * self.step_count
        elif self.model_architecture in ["aep", "preled"]:
            return self.step_size * 2
        else:
            return self.step_size

    @property
    def step_size(self) -> int:
        return self.config["step_size"]

    @property
    def step_count(self) -> int:
        return self.config["step_count"]

    @property
    def extra_steps(self) -> int:
        if "extra_steps" in self.config:
            return int(self.config["extra_steps"])
        return 0

    # endregion

    @property
    def channels(self) -> int:
        return self.config["channels"]

    # endregion

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
    def latent_code_size(self) -> int:
        return self.config["code_size"]

    @property
    def encoder_output_size(self) -> int:
        code_size = self.latent_code_size
        if self.model_architecture in ["vae", "viae", "vaegan", "iaegan", "avp"]:
            code_size *= 2
        return code_size

    @property
    def code_activation(self) -> str:
        return self.config["code_activation"]

    @property
    def kl_divergence_lambda(self) -> float:
        return self.get_config_value("kl_divergence_lambda", default=1e-2)

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

    # region Make models

    def make_ae(self) -> AE:
        encoder, decoder = self.make_encoder_decoder()
        model = AE(encoder=encoder, decoder=decoder)
        return model

    def make_vae(self) -> VAE:
        encoder, decoder = self.make_encoder_decoder()
        model = VAE(encoder=encoder,
                    decoder=decoder,
                    kl_divergence_loss_factor=self.kl_divergence_lambda,
                    )
        return model

    def make_iae(self) -> IAE:
        encoder, decoder = self.make_encoder_decoder()
        model = IAE(encoder=encoder,
                    decoder=decoder,
                    step_size=self.step_size,
                    use_stochastic_loss=False
                    )
        return model

    def make_viae(self) -> VIAE:
        encoder, decoder = self.make_encoder_decoder()
        model = VIAE(encoder=encoder,
                     decoder=decoder,
                     step_size=self.step_size,
                     use_stochastic_loss=False,
                     kl_divergence_lambda=self.kl_divergence_lambda,
                     )
        return model

    def make_iaegan(self) -> IAEGAN:
        encoder, decoder = self.make_encoder_decoder()
        discriminator = self.make_discriminator()

        model = IAEGAN(encoder=encoder,
                       decoder=decoder,
                       discriminator=discriminator,
                       step_size=self.step_size,
                       extra_steps=self.extra_steps,
                       use_stochastic_loss=False,
                       reconstruction_lambda=1e0,
                       kl_divergence_lambda=1e-3,
                       adversarial_lambda=1e-2,
                       gradient_penalty_lambda=1e1,
                       )
        return model

    def make_cnc(self) -> CnC:
        encoder, decoder = self.make_encoder_decoder()
        relevance_estimator = make_encoder(input_shape=self.encoder_input_shape,
                                           mode=self.encoder_mode,
                                           filters=self.encoder_filters,
                                           kernel_size=self.encoder_kernel_sizes,
                                           strides=self.encoder_strides,
                                           code_size=self.encoder_output_size,
                                           code_activation="linear",
                                           basic_block_count=self.basic_block_count,
                                           name="RelevanceEstimator",
                                           )

        model = CnC(encoder=encoder,
                    relevance_estimator=relevance_estimator,
                    decoder=decoder,
                    relevance_loss_weight=1.0,
                    skip_loss_weight=1.0,
                    energy_margin=1.0,
                    theta=0.5)
        return model

    def make_led(self) -> LED:
        encoder, decoder = self.make_encoder_decoder()

        # from custom_tf_models.description_length.LED import LEDGoal
        # goal_schedule = LEDGoal(initial_rate=0.03, decay_steps=1000, decay_rate=0.85, offset=0.03, staircase=False)
        goal_schedule = None

        model = LED(encoder=encoder,
                    decoder=decoder,
                    features_per_block=1,
                    merge_dims_with_features=True,
                    description_energy_loss_lambda=1e-3,
                    use_noise=True,
                    noise_stddev=0.1,
                    reconstruct_noise=False,
                    goal_schedule=goal_schedule,
                    allow_negative_description_loss_weight=True,
                    goal_delta_factor=4.0,
                    unmasked_reconstruction_weight=1.0,
                    energy_margin=1.0)
        return model

    def make_iterative_ae(self) -> IterativeAE:
        encoder_input_batch_shape = self.get_encoder_input_batch_shape()
        encoders = []
        decoders = []

        block_size = 16
        iteration_count = self.latent_code_size // block_size
        for i in range(iteration_count):
            encoder = make_encoder(input_shape=self.encoder_input_shape, mode=self.encoder_mode,
                                   filters=self.encoder_filters, kernel_size=self.encoder_kernel_sizes,
                                   strides=self.encoder_strides, code_size=block_size,
                                   code_activation=self.code_activation,
                                   basic_block_count=self.basic_block_count, name="Encoder_{}".format(i))
            decoder_input_shape = encoder.compute_output_shape(encoder_input_batch_shape)[1:]
            decoder = make_decoder(input_shape=decoder_input_shape, mode=self.decoder_mode,
                                   filters=self.decoder_filters, kernel_size=self.decoder_kernel_sizes,
                                   stem_kernel_size=self.stem_kernel_size, strides=self.decoder_strides,
                                   channels=self.channels, output_activation="linear",
                                   basic_block_count=self.basic_block_count, name="Decoder_{}".format(i))
            encoders.append(encoder)
            decoders.append(decoder)

        model = IterativeAE(encoders=encoders,
                            decoders=decoders,
                            output_activation=self.output_activation,
                            stop_accumulator_gradients=False,
                            )
        return model

    # endregion

    # region Make sub-models
    # region Base
    def make_encoder(self, input_shape, name="Encoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               mode=self.encoder_mode,
                               filters=self.encoder_filters,
                               kernel_size=self.encoder_kernel_sizes,
                               strides=self.encoder_strides,
                               code_size=self.encoder_output_size,
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

    def make_encoder_decoder(self, input_shape=None, encoder_name="Encoder", decoder_name="Decoder"):
        if input_shape is None:
            input_shape = self.encoder_input_shape

        encoder = self.make_encoder(input_shape, name=encoder_name)
        decoder = self.make_decoder(self.get_latent_code_shape(encoder), name=decoder_name)
        return encoder, decoder

    # endregion

    # region Adversarial
    def make_discriminator(self, input_shape=None) -> Model:
        if input_shape is None:
            input_shape = self.get_discriminator_input_shape()

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

    # region Sub-models input shapes

    def get_encoder_input_batch_shape(self, use_batch_size=False):
        batch_size = self.batch_size if use_batch_size else None
        shape = (batch_size, *self.encoder_input_shape)
        return shape

    def get_latent_code_batch_shape(self, encoder: Model):
        shape = encoder.compute_output_shape(self.get_encoder_input_batch_shape())
        shape = (*shape[:-1], self.latent_code_size)
        return shape

    def get_latent_code_shape(self, encoder: Model):
        return self.get_latent_code_batch_shape(encoder=encoder)[1:]

    def get_discriminator_input_shape(self) -> Tuple[int, int]:
        shape = self.encoder_input_shape
        if self.model_architecture in ["iaegan"]:
            shape = (self.input_length, shape[-1])
        return shape

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

    def setup_model(self, model: Model):
        model.build(self.get_encoder_input_batch_shape(False))
        if isinstance(model, (IAE, LED)):
            # noinspection PyProtectedMember
            model._set_inputs(tf.zeros(self.get_encoder_input_batch_shape(True)))
        model.compile(optimizer=self.make_base_optimizer())
