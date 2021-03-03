import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import Tuple, Callable

from protocols import DatasetProtocol, ProtocolTestConfig
from modalities import Pattern, ModalityLoadInfo, NetworkPacket
from custom_tf_models import AE, IAE
from custom_tf_models.adversarial import IAEGAN
from data_processing.network_packet_preprocessing import make_packets_augmentation, make_packets_preprocess
from callbacks.configs import NetworkPacketCallbackConfig


class PacketProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 base_log_dir: str,
                 epoch: int,
                 ):
        super(PacketProtocol, self).__init__(dataset_name=dataset_name,
                                             protocol_name="packet",
                                             base_log_dir=base_log_dir,
                                             epoch=epoch)

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        augment_network_packet = self.make_network_packet_augmentation()

        pattern = Pattern(
            ModalityLoadInfo(NetworkPacket, self.output_length),
            preprocessor=augment_network_packet
        )
        return pattern

    def get_network_packet_pattern(self) -> Pattern:
        network_packet_preprocess = self.make_network_packet_preprocess()

        pattern = Pattern(
            ModalityLoadInfo(NetworkPacket, self.output_length),
            preprocessor=network_packet_preprocess
        )
        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_network_packet_pattern().with_labels()

    # endregion

    @abstractmethod
    def get_test_config(self) -> ProtocolTestConfig:
        raise NotImplementedError

    # region Pre-processes / Post-process
    def make_network_packet_augmentation(self) -> Callable:
        return make_packets_augmentation(activation_range=self.output_activation,
                                         gaussian_noise_std=0.0)

    def make_network_packet_preprocess(self) -> Callable:
        return make_packets_preprocess(activation_range=self.output_activation)

    # endregion

    def make_model(self) -> Model:
        if self.model_architecture == "ae":
            model = self.make_ae()
        elif self.model_architecture == "iae":
            model = self.make_iae()
        elif self.model_architecture == "iaegan":
            model = self.make_iaegan()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

        self.setup_model(model)
        return model

    def setup_model(self, model: Model):
        model.build(self.get_encoder_input_batch_shape(False))
        if isinstance(model, IAE):
            # noinspection PyProtectedMember
            model._set_inputs(tf.zeros(self.get_encoder_input_batch_shape(True)))
        model.compile(optimizer=self.make_base_optimizer())

    # region Models

    def make_ae(self) -> AE:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = AE(encoder=encoder, decoder=decoder)
        return model

    def make_iae(self) -> IAE:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = IAE(encoder=encoder, decoder=decoder, step_size=self.step_size, use_stochastic_loss=False)
        return model

    def make_iaegan(self) -> IAEGAN:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        discriminator_input_shape = self.get_discriminator_input_shape()
        discriminator = self.make_discriminator(discriminator_input_shape)

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

    # endregion

    # region Sub-models input shapes
    def get_encoder_input_batch_shape(self, use_batch_size=False) -> Tuple[None, int, int]:
        batch_size = self.batch_size if use_batch_size else None
        shape = (batch_size, self.step_size, self.channels)
        return shape

    def get_encoder_input_shape(self) -> Tuple[int, int]:
        shape = self.get_encoder_input_batch_shape()[1:]
        return shape

    def get_latent_code_batch_shape(self, encoder: Model):
        shape = encoder.compute_output_shape(self.get_encoder_input_batch_shape())
        if self.model_architecture in ["iaegan"]:
            shape = (*shape[:-1], shape[-1] // 2)
        return shape

    def get_latent_code_shape(self, encoder: Model):
        return self.get_latent_code_batch_shape(encoder=encoder)[1:]

    def get_discriminator_input_shape(self) -> Tuple[int, int]:
        shape = self.get_encoder_input_shape()
        if self.model_architecture in ["iaegan"]:
            shape = (self.input_length, shape[-1])
        return shape

    # endregion

    # region Callbacks
    def get_modality_callback_configs(self):
        pattern = self.get_network_packet_pattern()
        configs = []

        model = self.model
        if isinstance(model, AE):
            configs += [
                NetworkPacketCallbackConfig(autoencoder=model, pattern=pattern, is_train_callback=True,
                                            name="train"),
            ]

        if isinstance(model, IAE):
            config = NetworkPacketCallbackConfig(autoencoder=model.interpolate, pattern=pattern,
                                                 is_train_callback=True, name="interpolate_test")
            configs.append(config)

        return configs

    # endregion

    # region Properties

    @property
    def input_length(self) -> int:
        if self.model_architecture in ["aep", "preled"]:
            return self.step_size
        else:
            return self.output_length

    @property
    def output_length(self) -> int:
        if self.model_architecture in ["iae", "and", "iaegan", "avp"]:
            return self.step_size * self.step_count
        elif self.model_architecture in ["aep", "preled"]:
            return self.step_size * 2
        else:
            return self.step_size

    # region Config properties
    @property
    def step_size(self) -> int:
        return self.config["step_size"]

    @property
    def step_count(self) -> int:
        return self.config["step_count"]

    @property
    def extra_steps(self) -> int:
        return 0
    # endregion
    # endregion
