from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import Callable, Dict, Optional

from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols import AUCCallbackConfig
from modalities import Pattern
from models import IAE


class DatasetProtocol(Protocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 model_name: str = None
                 ):
        model = self.make_model()
        autoencoder = self.make_autoencoder()

        super(DatasetProtocol, self).__init__(model=model,
                                              dataset_name=dataset_name,
                                              protocol_name=protocol_name,
                                              autoencoder=autoencoder,
                                              model_name=model_name)

    # region Init
    @abstractmethod
    def make_model(self) -> Model:
        raise NotImplementedError

    @staticmethod
    def make_autoencoder() -> Optional[Callable]:
        return None
    # endregion

    # region Train
    def train_model(self, config: ProtocolTrainConfig = None):
        if config is None:
            config = self.get_train_config()

        super(DatasetProtocol, self).train_model(config)

    @abstractmethod
    def get_train_config(self) -> ProtocolTrainConfig:
        raise NotImplementedError
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

    def get_auc_callbacks_configs(self):
        anomaly_pattern = self.get_anomaly_pattern()

        auc_callbacks_configs = [
            AUCCallbackConfig(self.model, anomaly_pattern, self.output_length, prefix=""),

        ]

        if isinstance(self.model, IAE):
            auc_callbacks_configs += \
                [AUCCallbackConfig(self.model.interpolate, anomaly_pattern, self.output_length, prefix="iae")]

    @property
    @abstractmethod
    def output_length(self) -> int:
        raise NotImplementedError

    # region Utility
    @abstractmethod
    def get_config(self) -> Dict:
        return {}
    # endregion
