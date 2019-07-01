import os
import csv
from tqdm import tqdm
from typing import Dict, Tuple, List, Union, Optional

from modalities import ModalityCollection
from datasets.tfrecord_builders import TFRecordBuilder, DataSource


class EmolyTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 video_frequency: Optional[Union[int, float]],
                 audio_frequency: Optional[Union[int, float]],
                 modalities: ModalityCollection,
                 video_frame_size: Tuple[int, int],
                 verbose=1):
        super(EmolyTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                   shard_duration=shard_duration,
                                                   video_frequency=video_frequency,
                                                   audio_frequency=audio_frequency,
                                                   modalities=modalities,
                                                   verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_dataset_sources(self) -> List[DataSource]:
        video_filenames = self.list_videos_filenames()
        labels = self.get_labels()

        data_sources: List[DataSource] = []
        for video_filename in video_filenames:
            sample_name = video_filename[:-4]
            if sample_name in labels:
                sample_labels = labels[sample_name]
                is_train_sample = False
            else:
                sample_labels = [(0.0, 0.0)]
                is_train_sample = True

            video_path = os.path.join(self.videos_folder, video_filename)
            subset_name = "Train" if is_train_sample else "Test"
            target_path = os.path.join(self.dataset_path, subset_name, sample_name)

            data_source = DataSource(labels_source=sample_labels,
                                     target_path=target_path,
                                     subset_name=subset_name,
                                     video_source=video_path,
                                     video_frame_size=self.video_frame_size,
                                     audio_source=video_path)
            data_sources.append(data_source)

        return data_sources

    def list_videos_filenames(self):
        elements = os.listdir(self.videos_folder)
        videos_filenames = []
        for video_filename in elements:
            if video_filename.endswith(".mp4") and os.path.isfile(os.path.join(self.videos_folder, video_filename)):
                videos_filenames.append(video_filename)
        return videos_filenames

    def rename_videos(self):
        video_filenames = self.list_videos_filenames()
        for video_index in tqdm(range(len(video_filenames))):
            video_filename = video_filenames[video_index]
            video_filepath = os.path.join(self.videos_folder, video_filename)

            if "actee" in video_filename:
                target_filepath = video_filepath.replace("actee", "acted")
                os.rename(video_filepath, target_filepath)
            elif "induit" in video_filename:
                target_filepath = video_filepath.replace("induit", "induced")
                os.rename(video_filepath, target_filepath)

    def get_labels(self) -> Dict[str, List[Tuple[float, float]]]:
        strength_ids = {"absent": 0, "trace": 1, "light": 2, "marked": 3, "severe": 4, "maximum": 5}
        strength_split_value = strength_ids["trace"]
        labels: Dict[str, List[Tuple[float, float]]] = {}

        with open(self.labels, 'r') as labels_file:
            reader = csv.reader(labels_file, delimiter=',')
            for row in reader:
                assert len(row) == 4

                sample, start, end, strength = row

                assert len(sample) > 0

                if sample == "Sample":
                    continue

                if len(start) == 0 or len(end) == 0 or strength not in strength_ids:
                    start = 0.0
                    end = 0.0
                else:
                    strength = strength_ids[strength]
                    if strength <= strength_split_value:
                        start = 0.0
                        end = 0.0
                    else:
                        start = float(start)
                        end = float(end)

                sample_labels = (start, end)
                if sample in labels:
                    labels[sample].append(sample_labels)
                else:
                    labels[sample] = [sample_labels]
        return labels

    @staticmethod
    def split_labels_by_strength(labels: Dict[str, Tuple[int, int, int]]
                                 ) -> Dict[int, Dict[str, Tuple[int, int]]]:
        output_labels: Dict[int, Dict[str, Tuple[int, int]]] = {}

        for sample, (start, end, strength) in labels.items():
            if strength in output_labels:
                output_labels[strength][sample] = (start, end)
            else:
                output_labels[strength] = {sample: (start, end)}

        return output_labels

    @property
    def videos_folder(self):
        return os.path.join(self.dataset_path, "video")

    @property
    def labels(self):
        return os.path.join(self.dataset_path, "labels_en.csv")


def main():
    # from modalities import RawVideo
    # from modalities import OpticalFlow
    # from modalities import DoG
    # from modalities import RawAudio
    # from modalities import MelSpectrogram
    from modalities import Landmarks

    emoly_tf_record_builder = EmolyTFRecordBuilder(dataset_path="../datasets/emoly",
                                                   shard_duration=1.28,
                                                   video_frequency=25,
                                                   audio_frequency=48000,
                                                   modalities=ModalityCollection(
                                                       [
                                                           # RawVideo(),
                                                           # OpticalFlow(use_polar=False),
                                                           # DoG(),
                                                           # RawAudio(),
                                                           # MelSpectrogram(window_width=0.03,
                                                           #                window_step=0.015,
                                                           #                mel_filters_count=40),
                                                           Landmarks("../shape_predictor_68_face_landmarks.dat")
                                                       ]
                                                   ),
                                                   video_frame_size=(1080//4, 1920//4))
    emoly_tf_record_builder.build()


if __name__ == "__main__":
    main()
