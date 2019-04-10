import os
import csv
from tqdm import tqdm
from typing import Dict, Tuple, Any, List

from datasets.tfrecord_builders import TFRecordBuilder, DataSource


class EmolyTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 modalities: Dict[str, Dict[str, Any]],
                 video_frame_size: Tuple[int, int],
                 verbose=1):
        super(EmolyTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                   shard_duration=shard_duration,
                                                   modalities=modalities,
                                                   verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_dataset_sources(self) -> List[DataSource]:
        raise NotImplementedError

    def list_videos_filenames(self):
        elements = os.listdir(self.videos_folder)
        videos_filenames = []
        for video_filename in elements:
            if ".mp4" in video_filename and os.path.isfile(os.path.join(self.videos_folder, video_filename)):
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

    def get_labels(self) -> Dict[str, Tuple[int, int, int]]:
        strength_ids = {"absent": 0, "trace": 1, "light": 2, "marked": 3, "severe": 4, "maximum": 5}
        labels = {}

        with open(self.labels, 'r') as labels_file:
            reader = csv.reader(labels_file, delimiter=',')
            for row in reader:
                assert len(row) == 4

                sample, start, end, strength = row

                assert len(sample) > 0

                if sample == "Sample":
                    continue

                if len(start) == 0 or len(end) == 0 or strength not in strength_ids:
                    start = 0
                    end = 0
                    strength = strength_ids["absent"]
                else:
                    start = int(25.0 * float(start))
                    end = int(25.0 * float(end))
                    strength = strength_ids[strength]

                labels[sample] = (start, end, strength)
        return labels

    @property
    def videos_folder(self):
        return os.path.join(self.dataset_path, "video")

    @property
    def labels(self):
        return os.path.join(self.dataset_path, "labels_en.csv")


if __name__ == "__main__":
    emoly_tf_record_builder = EmolyTFRecordBuilder(dataset_path="../datasets/emoly",
                                                   shard_duration=2.0,
                                                   modalities=
                                                   {
                                                       "raw_video":
                                                           {
                                                               "frequency": 25
                                                           },
                                                       "flow":
                                                           {
                                                               "use_polar": True,
                                                               "frequency": "raw_video"
                                                           },
                                                       "dog":
                                                           {
                                                               "frequency": "raw_video"
                                                           }
                                                   },
                                                   video_frame_size=(128, 128))
    emoly_tf_record_builder.build()
