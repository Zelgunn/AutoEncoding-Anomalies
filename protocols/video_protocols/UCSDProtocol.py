from typing import Dict

from protocols import VideoProtocol, ProtocolTestConfig


class UCSDProtocol(VideoProtocol):
    def __init__(self,
                 base_log_dir: str,
                 dataset_version=2,
                 epoch=0,
                 config: Dict = None,
                 ):
        if dataset_version not in [1, 2]:
            raise ValueError("`dataset_version` must either be 1 (ped1) or 2 (ped2). Received {}."
                             .format(dataset_version))
        dataset_name = "ped1" if dataset_version == 1 else "ped2"

        super(UCSDProtocol, self).__init__(dataset_name=dataset_name,
                                           epoch=epoch,
                                           base_log_dir=base_log_dir,
                                           config=config)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()

        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.epoch,
                                  detector_stride=1,
                                  pre_normalize_predictions=True,
                                  compare_metrics=["mae"]
                                  )

    @property
    def dataset_channels(self) -> int:
        return 1

    @property
    def use_face(self) -> bool:
        return False

    @property
    def video_sample_rate(self) -> int:
        return 10
