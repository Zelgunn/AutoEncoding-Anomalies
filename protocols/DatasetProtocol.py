from tensorflow.python.keras import Model
import numpy as np
from abc import abstractmethod
from typing import Callable, Dict, Optional, List
import json
import os

from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from callbacks.configs import AUCCallbackConfig
from modalities import Pattern
from models import IAE


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
                                   modality_callback_configs=modality_callback_configs,
                                   auc_callback_configs=auc_callbacks_configs,
                                   early_stopping_metric=self.model.metrics_names[0])

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

    def get_auc_callbacks_configs(self) -> List[AUCCallbackConfig]:
        anomaly_pattern = self.get_anomaly_pattern()

        auc_callbacks_configs = [
            AUCCallbackConfig(self.model, anomaly_pattern, self.output_length, prefix=""),

        ]

        if isinstance(self.model, IAE):
            auc_callbacks_configs += \
                [AUCCallbackConfig(self.model.interpolate, anomaly_pattern, self.output_length, prefix="iae")]

        return auc_callbacks_configs

    def make_log_dir(self, sub_folder: str) -> str:
        log_dir = super(DatasetProtocol, self).make_log_dir(sub_folder)
        self.save_config(log_dir)
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

    def save_config(self, log_dir: str):
        config_path = os.path.join(log_dir, "main_config.json")
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file)

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
    def batch_size(self) -> int:
        return self.config["batch_size"]

    @property
    def output_activation(self) -> str:
        return self.config["output_activation"]
    # endregion
