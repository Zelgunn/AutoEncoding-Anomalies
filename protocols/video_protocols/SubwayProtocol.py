from datasets.tfrecord_builders.SubwayTFRB import SubwayVideo
from protocols import VideoProtocol, ProtocolTestConfig


class SubwayProtocol(VideoProtocol):
    def __init__(self,
                 video_id: SubwayVideo,
                 initial_epoch=0,
                 model_name=None
                 ):
        if video_id == SubwayVideo.EXIT:
            dataset_name = "subway_exit"
        elif video_id == SubwayVideo.ENTRANCE:
            dataset_name = "subway_entrance"
        elif video_id == SubwayVideo.MALL1:
            dataset_name = "subway_mall1"
        elif video_id == SubwayVideo.MALL1:
            dataset_name = "subway_mall2"
        elif video_id == SubwayVideo.MALL1:
            dataset_name = "subway_mall3"
        else:
            raise ValueError("Error : {} is not a valid Subway video ID.".format(video_id))

        super(SubwayProtocol, self).__init__(dataset_name=dataset_name,
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
        return False

    @property
    def video_sample_rate(self) -> int:
        return 25
