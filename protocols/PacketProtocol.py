from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import Tuple, Callable, Dict, List

from protocols import DatasetProtocol, ProtocolTestConfig
from modalities import Pattern, ModalityLoadInfo, NetworkPacket
from custom_tf_models import AE, IAE, LED
from data_processing.network_packet_preprocessing import make_packets_augmentation, make_packets_preprocess
from callbacks.configs import NetworkPacketCallbackConfig, AUCCallbackConfig


class PacketProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 base_log_dir: str,
                 epoch: int,
                 config: Dict = None,
                 ):
        super(PacketProtocol, self).__init__(dataset_name=dataset_name,
                                             protocol_name="packet",
                                             base_log_dir=base_log_dir,
                                             epoch=epoch,
                                             config=config, )

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
                                         gaussian_noise_std=float(self.config["gaussian_noise_std"]))

    def make_network_packet_preprocess(self) -> Callable:
        return make_packets_preprocess(activation_range=self.output_activation)

    # endregion

    def make_model(self) -> Model:
        if self.model_architecture == "ae":
            model = self.make_ae()
        elif self.model_architecture == "vae":
            model = self.make_vae()
        elif self.model_architecture == "iae":
            model = self.make_iae()
        elif self.model_architecture == "viae":
            model = self.make_viae()
        elif self.model_architecture == "iaegan":
            model = self.make_iaegan()
        elif self.model_architecture == "cnc":
            model = self.make_cnc()
        elif self.model_architecture == "led":
            model = self.make_led()
        elif self.model_architecture == "iterative_ae":
            model = self.make_iterative_ae()
        elif self.model_architecture == "ltm":
            model = self.make_ltm()
        else:
            raise ValueError("Unknown architecture : {}".format(self.model_architecture))

        self.setup_model(model)
        return model

    # region Callbacks
    def get_modality_callback_configs(self):
        train_pattern = self.get_train_pattern()
        test_pattern = self.get_network_packet_pattern()
        configs = []

        model = self.model
        if isinstance(model, AE):
            configs += [
                NetworkPacketCallbackConfig(autoencoder=model, pattern=train_pattern, is_train_callback=True,
                                            name="train"),
                NetworkPacketCallbackConfig(autoencoder=model, pattern=test_pattern, is_train_callback=False,
                                            name="test"),
            ]

        if isinstance(model, IAE):
            config = NetworkPacketCallbackConfig(autoencoder=model.interpolate, pattern=test_pattern,
                                                 is_train_callback=False, name="interpolate_test")
            configs.append(config)

        return configs

    def get_auc_callback_configs(self) -> List[AUCCallbackConfig]:
        if self.auc_frequency < 1:
            return []

        auc_callbacks_configs = super(PacketProtocol, self).get_auc_callback_configs()
        anomaly_pattern = self.get_anomaly_pattern()
        if isinstance(self.model, LED):
            auc_callbacks_configs += [
                AUCCallbackConfig(self.model.compute_description_energy, anomaly_pattern, labels_length=1, prefix="LED",
                                  convert_to_io_compare_model=False, epoch_freq=self.auc_frequency,
                                  sample_count=self.auc_sample_count)
            ]

        return auc_callbacks_configs

    def get_anomaly_detector_callback_configs(self):
        return []

    # endregion

    @property
    def encoder_input_shape(self) -> Tuple[int, int]:
        return self.step_size, self.channels
