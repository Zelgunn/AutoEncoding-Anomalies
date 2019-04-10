import os
from typing import Tuple, List, Dict, Any

from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.modality_builders import VideoReader


class UCSDTFRecordBuilder(TFRecordBuilder):
    def __init__(self,
                 dataset_path: str,
                 shard_duration: float,
                 modalities: Dict[str, Dict[str, Any]],
                 video_frame_size: Tuple[int, int],
                 verbose=1):
        super(UCSDTFRecordBuilder, self).__init__(dataset_path=dataset_path,
                                                  shard_duration=shard_duration,
                                                  modalities=modalities,
                                                  verbose=verbose)
        self.video_frame_size = video_frame_size

    def get_dataset_sources(self) -> List[DataSource]:
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
                                   subset_name=subset,
                                   video_source=VideoReader(path),
                                   video_frame_size=self.video_frame_size)
                        for subset in subsets
                        for path, labels in subsets[subset]]
        return data_sources


if __name__ == "__main__":
    ucsd_tf_record_builder = UCSDTFRecordBuilder(dataset_path="../datasets/ucsd/ped2",
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
                                                 video_frame_size=(128, 128)
                                                 )
    ucsd_tf_record_builder.build()
