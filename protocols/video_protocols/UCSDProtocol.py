from protocols import VideoProtocol, ProtocolTestConfig


class UCSDProtocol(VideoProtocol):
    def __init__(self,
                 dataset_version=2,
                 initial_epoch=0,
                 model_name=None
                 ):
        if dataset_version not in [1, 2]:
            raise ValueError("`dataset_version` must either be 1 (ped1) or 2 (ped2). Received {}."
                             .format(dataset_version))
        dataset_name = "ped1" if dataset_version == 1 else "ped2"

        super(UCSDProtocol, self).__init__(dataset_name=dataset_name,
                                           initial_epoch=initial_epoch,
                                           model_name=model_name)

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()

        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.initial_epoch,
                                  detector_stride=1,
                                  pre_normalize_predictions=True
                                  )

    @property
    def dataset_channels(self) -> int:
        return 1

    @property
    def use_face(self) -> bool:
        return False
