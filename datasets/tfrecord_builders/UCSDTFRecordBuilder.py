from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from typing import Type, Tuple

from datasets.tfrecord_builders import TFRecordBuilder, DataSource
from datasets.modality_builders import VideoReader


class UCSDTFRecordBuilder(TFRecordBuilder):
    def build_datasets(self, shard_size: int, video_frame_size: Tuple[int, int]):
        subsets = {"Test": 12, "Train": 16}
        subsets = {subset: [os.path.join(self.dataset_path, "{}{:03d}".format(subset, i + 1))
                            for i in range(subsets[subset])]
                   for subset in subsets}
        test_labels = ["{}_gt".format(path) for path in subsets["Test"]]
        train_labels = [False for _ in subsets["Train"]]
        labels = {"Test": test_labels, "Train": train_labels}
        data_sources = [DataSource(labels_source=labels[subset],
                                   target_path=None,
                                   video_source=VideoReader(subsets[subset]))
                        for subset in subsets]

        for data_source in data_sources:
            shard_count = int(np.ceil(data_source.video_source.frame_count / shard_size))
            self.build(shard_count, data_source)


def get_videos_dictionary(subset_path: str, extension: str):
    subset_path = os.path.normpath(subset_path)
    if not extension.startswith("."):
        extension = "." + extension
    sub_directories = [dir_info[0] for dir_info in os.walk(subset_path)]
    if subset_path in sub_directories:
        sub_directories.remove(subset_path)
    tiff_images_dictionary = {}
    for sub_directory in sub_directories:
        tiff_images_paths = get_files_with_extension(sub_directory, extension)
        if len(tiff_images_paths) is not 0:
            tiff_images_dictionary[sub_directory] = tiff_images_paths
    return tiff_images_dictionary


def get_files_with_extension(directory: str, extension: str, sort=True):
    files = os.listdir(directory)
    result = []
    for file in files:
        if file.endswith(extension):
            result += [file]
    if sort:
        result.sort()
    return result


def load_videos_from_dictionary_of_paths(dictionary, dtype: Type = np.float32):
    videos = []
    sub_directories = dictionary.keys()
    sub_directories = sorted(sub_directories)
    for sub_directory in sub_directories:
        tiff_images_paths = dictionary[sub_directory]
        video_length = len(tiff_images_paths)
        sub_directory_info: str = sub_directory.split(os.path.sep)[-1]
        video = None
        for i in tqdm(range(video_length), desc=sub_directory_info):
            image_path = os.path.join(sub_directory, tiff_images_paths[i])
            image = Image.open(image_path)
            image = np.array(image)
            if video is None:
                video = np.empty(shape=[video_length, image.shape[0], image.shape[1]], dtype=dtype)
            video[i] = image
        video = np.expand_dims(video, axis=-1)
        videos.append(video)
    return videos


def load_ucsd_videos_raw(subset_path: str):
    videos_dictionary = get_videos_dictionary(subset_path, ".tif")
    return load_videos_from_dictionary_of_paths(videos_dictionary)


if __name__ == "__main__":
    ucsd_tf_record_builder = UCSDTFRecordBuilder(dataset_path="../datasets/ucsd",
                                                 modalities={"raw_video": {},
                                                             "flow": {"use_polar": True},
                                                             "dog": {}
                                                             }
                                                 )
    ucsd_tf_record_builder.build_datasets()
