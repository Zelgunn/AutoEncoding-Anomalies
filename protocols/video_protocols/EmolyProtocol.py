from protocols import VideoProtocol, ProtocolTestConfig


class EmolyProtocol(VideoProtocol):
    def __init__(self,
                 initial_epoch=0,
                 model_name=None
                 ):
        super(EmolyProtocol, self).__init__(dataset_name="emoly",
                                            initial_epoch=initial_epoch,
                                            model_name=model_name)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.initial_epoch,
                                  detector_stride=32,
                                  pre_normalize_predictions=True)

    @property
    def dataset_channels(self) -> int:
        return 3

    @property
    def use_face(self) -> bool:
        return True
