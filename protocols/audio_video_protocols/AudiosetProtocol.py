from typing import Dict

from protocols import AudioVideoProtocol, ProtocolTestConfig


class AudiosetProtocol(AudioVideoProtocol):
    def __init__(self,
                 base_log_dir: str,
                 epoch=0,
                 config: Dict = None,
                 ):
        super(AudiosetProtocol, self).__init__(base_log_dir=base_log_dir,
                                               dataset_name="audioset",
                                               epoch=epoch,
                                               config=config)

    def get_test_config(self) -> ProtocolTestConfig:
        raise RuntimeError("Audioset is not an anomaly detection dataset.")

    @property
    def video_sample_rate(self) -> int:
        return 25

    @property
    def audio_sample_rate(self) -> int:
        if self.use_mfcc:
            return 100
        else:
            return 48000
