from typing import Dict


from protocols import AudioVideoProtocol, ProtocolTestConfig


class EmolyProtocol(AudioVideoProtocol):
    def __init__(self,
                 base_log_dir: str,
                 epoch=0,
                 config: Dict = None,
                 ):
        super(EmolyProtocol, self).__init__(base_log_dir=base_log_dir,
                                            dataset_name="emoly",
                                            epoch=epoch,
                                            config=config)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.epoch,
                                  detector_stride=32,
                                  pre_normalize_predictions=False)

    @property
    def video_sample_rate(self) -> int:
        return 25

    @property
    def audio_sample_rate(self) -> int:
        if self.use_mfcc:
            return 100
        else:
            return 48000
