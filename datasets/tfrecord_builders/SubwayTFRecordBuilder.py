import os
from typing import Tuple

from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.modality_builders import VideoReader


class SubwayTFRecordBuilder(TFRecordBuilder):
    def build_datasets(self, video_frame_size: Tuple[int, int]):
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
                                       video_source=train_video_reader,
                                       video_frame_size=video_frame_size)

        test_video_reader = VideoReader(video_filepath, end=-training_frames)
        test_labels = [(40880, 41160), (41400, 41700), (50410, 50710), (50980, 51250), (60160, 60940)]
        test_labels = [(start-training_frames, end-training_frames) for start, end in test_labels]
        test_target_path = os.path.join(self.dataset_path, "Test")
        if not os.path.isdir(test_target_path):
            os.makedirs(test_target_path)
        test_data_source = DataSource(labels_source=test_labels,
                                      target_path=test_target_path,
                                      video_source=test_video_reader,
                                      video_frame_size=video_frame_size)

        data_sources = [train_data_source, test_data_source]
        self.build(data_sources)


if __name__ == "__main__":
    subway_tf_record_builder = SubwayTFRecordBuilder(dataset_path="../datasets/subway/exit",
                                                     modalities=
                                                     {
                                                         "raw_video":
                                                             {
                                                                 "shard_size": 32
                                                             },
                                                         "flow":
                                                             {
                                                                 "use_polar": True,
                                                                 "shard_size": "raw_video"
                                                             },
                                                         "dog":
                                                             {
                                                                 "shard_size": "raw_video"
                                                             }
                                                     }
                                                     )
    subway_tf_record_builder.build_datasets(video_frame_size=(128, 128))
