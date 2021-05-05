from typing import Dict

from protocols import VideoProtocol, ProtocolTestConfig


class ShanghaiTechProtocol(VideoProtocol):
    def __init__(self,
                 base_log_dir: str,
                 epoch=0,
                 config: Dict = None,
                 ):
        super(ShanghaiTechProtocol, self).__init__(base_log_dir=base_log_dir,
                                                   dataset_name="shanghaitech",
                                                   epoch=epoch,
                                                   config=config)

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
        return False

    @property
    def video_sample_rate(self) -> int:
        return 25
