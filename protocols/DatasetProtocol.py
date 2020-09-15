from tensorflow.python.keras.models import Model
import numpy as np
from abc import abstractmethod
from typing import Callable, Dict, Optional, List, Union
import json
import os
from shutil import copyfile

from datasets.tfrecord_builders import tfrecords_config_filename
from modalities import Pattern
from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from callbacks.configs import AUCCallbackConfig
from custom_tf_models import AE, IAE, LED
from custom_tf_models.energy_based import EBAE


class DatasetProtocol(Protocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 initial_epoch: int,
                 model_name: str = None
                 ):

        self.config = self.load_config(protocol_name, dataset_name)
        if "seed" not in self.config:
            self.config["seed"] = int(np.random.randint(low=0, high=2 ** 31, dtype=np.int32))
        self.seed = self.config["seed"]

        model = self.make_model()
        autoencoder = self.make_autoencoder(model)
        self.initial_epoch = initial_epoch

        output_range = (-1.0, 1.0) if self.output_activation == "tanh" else (0.0, 1.0)

        super(DatasetProtocol, self).__init__(model=model,
                                              dataset_name=dataset_name,
                                              protocol_name=protocol_name,
                                              autoencoder=autoencoder,
                                              model_name=model_name,
                                              output_range=output_range,
                                              seed=self.seed)

    # region Init
    @abstractmethod
    def make_model(self) -> Model:
        raise NotImplementedError

    @staticmethod
    def make_autoencoder(model: Model) -> Optional[Callable]:
        return model

    # endregion

    # region Train
    def train_model(self, config: ProtocolTrainConfig = None, **kwargs):
        if config is None:
            config = self.get_train_config()

        super(DatasetProtocol, self).train_model(config)

    def get_train_config(self) -> ProtocolTrainConfig:
        train_pattern = self.get_train_pattern()
        modality_callback_configs = self.get_modality_callback_configs()
        auc_callbacks_configs = self.get_auc_callbacks_configs()

        return ProtocolTrainConfig(batch_size=self.batch_size,
                                   pattern=train_pattern,
                                   steps_per_epoch=self.steps_per_epoch,
                                   epochs=self.epochs,
                                   initial_epoch=self.initial_epoch,
                                   validation_steps=self.validation_steps,
                                   save_frequency=self.save_frequency,
                                   modality_callback_configs=modality_callback_configs,
                                   auc_callback_configs=auc_callbacks_configs)

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
    def get_auc_callbacks_configs(self) -> List[AUCCallbackConfig]:
        anomaly_pattern = self.get_anomaly_pattern()
        auc_callbacks_configs = []

        model = self.model
        if isinstance(model, EBAE):
            model = model.autoencoder

        if isinstance(model, AE):
            auc_callbacks_configs += [
                AUCCallbackConfig(model, anomaly_pattern, labels_length=self.output_length, prefix="",
                                  convert_to_io_compare_model=True),
            ]

        if isinstance(model, IAE):
            auc_callbacks_configs += \
                [AUCCallbackConfig(model.interpolate, anomaly_pattern, labels_length=self.output_length,
                                   prefix="iae", convert_to_io_compare_model=True)
                 ]

        if isinstance(model, LED):
            auc_callbacks_configs += \
                [AUCCallbackConfig(model.compute_description_energy, anomaly_pattern, labels_length=1,
                                   prefix="desc_energy", epoch_freq=1)
                 ]

        return auc_callbacks_configs

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
    # endregion
