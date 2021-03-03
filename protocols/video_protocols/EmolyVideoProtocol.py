from protocols import VideoProtocol, ProtocolTestConfig


class EmolyVideoProtocol(VideoProtocol):
    def __init__(self,
                 base_log_dir: str,
                 epoch=0,
                 ):
        super(EmolyVideoProtocol, self).__init__(base_log_dir=base_log_dir,
                                                 dataset_name="emoly",
                                                 epoch=epoch)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.epoch,
                                  detector_stride=32,
                                  pre_normalize_predictions=True)

    @property
    def dataset_channels(self) -> int:
        return 3

    @property
    def use_face(self) -> bool:
        return True

    @property
    def video_sample_rate(self) -> int:
        return 25
