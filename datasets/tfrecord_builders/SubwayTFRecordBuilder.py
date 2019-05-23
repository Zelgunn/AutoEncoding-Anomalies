import os
from typing import Tuple, List, Union

from modalities import ModalityCollection, RawVideo, OpticalFlow, DoG
from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.data_readers import VideoReader


class SubwayTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Union[int, float],
                 audio_frequency: Union[int, float],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 verbose=1):
        super(SubwayTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                    shard_duration=shard_duration,
                                                    video_frequency=video_frequency,
                                                    audio_frequency=audio_frequency,
                                                    modalities=modalities,
                                                    verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_dataset_sources(self) -> List[DataSource]:
        video_filename = "Subway_Exit.avi"
        video_filepath = os.path.join(self.dataset_path, video_filename)
        fps = 25.0
        training_minutes = 10.0
        training_frames = int(fps * training_minutes * 60)

        train_video_reader = VideoReader(video_filepath, end=training_frames)
        train_labels = False
        train_target_path = os.path.join(self.dataset_path, "Train")
        if not os.path.isdir(train_target_path):
            os.makedirs(train_target_path)
        train_data_source = DataSource(labels_source=train_labels,
                                       target_path=train_target_path,
                                       subset_name="Train",
                                       video_source=train_video_reader,
                                       video_frame_size=self.video_frame_size)

        test_video_reader = VideoReader(video_filepath, start=training_frames)
        test_labels = [(40880, 41160), (41400, 41700), (50410, 50710), (50980, 51250), (60160, 60940)]
        test_labels = [((start - training_frames) / fps, (end - training_frames) / fps)
                       for start, end in test_labels]
        test_target_path = os.path.join(self.dataset_path, "Test")
        if not os.path.isdir(test_target_path):
            os.makedirs(test_target_path)
        test_data_source = DataSource(labels_source=test_labels,
                                      target_path=test_target_path,
                                      subset_name="Test",
                                      video_source=test_video_reader,
                                      video_frame_size=self.video_frame_size)

        data_sources = [train_data_source, test_data_source]
        return data_sources


if __name__ == "__main__":
    subway_tf_record_builder = SubwayTFRecordBuilder(dataset_path="../datasets/subway/exit",
                                                     shard_duration=1.28,
                                                     modalities=ModalityCollection(
                                                         [
                                                             RawVideo(frequency=25),
                                                             OpticalFlow(frequency=25, use_polar=True),
                                                             DoG(frequency=25),
                                                         ]
                                                     ),
                                                     video_frame_size=(128, 128)
                                                     )
    subway_tf_record_builder.build()
