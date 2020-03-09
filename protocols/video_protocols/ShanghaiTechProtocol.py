from protocols import VideoProtocol, ProtocolTestConfig


class ShanghaiTechProtocol(VideoProtocol):
    def __init__(self,
                 initial_epoch=0,
                 model_name=None
                 ):
        super(ShanghaiTechProtocol, self).__init__(dataset_name="shanghaitech",
                                                   initial_epoch=initial_epoch,
                                                   model_name=model_name)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.initial_epoch,
                                  detector_stride=1,
                                  pre_normalize_predictions=True)

    @property
    def dataset_channels(self) -> int:
        return 3

    @property
    def use_face(self) -> bool:
        return False

    @property
    def video_sample_rate(self) -> int:
        return 25
