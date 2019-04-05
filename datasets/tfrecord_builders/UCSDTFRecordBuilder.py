import os
from typing import Tuple

from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.modality_builders import VideoReader


class UCSDTFRecordBuilder(TFRecordBuilder):
    def build_datasets(self, video_frame_size: Tuple[int, int]):
        subsets_lengths = {"Test": 12, "Train": 16}
        subsets = {}
        for subset in subsets_lengths:
            paths = []
            for i in range(subsets_lengths[subset]):
                filename = "{subset}/{subset}{index:03d}".format(subset=subset, index=i + 1)
                path = os.path.join(self.dataset_path, filename)
                path = os.path.normpath(path)
                paths.append(path)
            subsets[subset] = paths

        test_labels = ["{}_gt".format(path) for path in subsets["Test"]]
        train_labels = [False for _ in subsets["Train"]]
        labels = {"Test": test_labels, "Train": train_labels}

        subsets = {subset: zip(subsets[subset], labels[subset]) for subset in subsets}

        data_sources = [DataSource(labels_source=labels,
                                   target_path=path,
                                   video_source=VideoReader(path),
                                   video_frame_size=video_frame_size)
                        for subset in subsets
                        for path, labels in subsets[subset]]

        self.build(data_sources)


if __name__ == "__main__":
    ucsd_tf_record_builder = UCSDTFRecordBuilder(dataset_path="../datasets/ucsd/ped2",
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
    ucsd_tf_record_builder.build_datasets(video_frame_size=(128, 128))
