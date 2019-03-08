import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Type

from datasets import FullyLoadableDataset


class UCSDDataset(FullyLoadableDataset):
    # region Loading
    def load(self):
        if self.dataset_path.endswith(os.path.sep):
            self.dataset_path = self.dataset_path[:-1]

        self.pixel_labels = None

        self.saved_to_npz = os.path.exists(self.npz_filepath)
        print("Loading UCSD dataset : " + self.dataset_path)
        if self.saved_to_npz:
            print("Found saved dataset (in .npz file)")
            npz_file = np.load(self.npz_filepath)
            self.videos = npz_file["videos"]
            if "anomaly_labels" in npz_file:
                self.pixel_labels = npz_file["anomaly_labels"]
                if None in self.pixel_labels:
                    self.pixel_labels = None
        else:
            print("Building dataset from raw data")
            self.videos = UCSDDataset._load_ucsd_videos_raw(self.dataset_path)
            labels_dictionary = UCSDDataset._get_videos_dictionary(self.dataset_path, ".bmp")
            if len(labels_dictionary) > 0:
                self.pixel_labels = UCSDDataset._load_videos_from_dictionary_of_paths(labels_dictionary, dtype=np.bool)
            else:
                self.pixel_labels = None

    def save_to_npz(self, force=False):
        if self.saved_to_npz and not force:
            return
        print("Saving UCSD dataset : " + self.dataset_path + " (to .npz file)")
        np.savez(self.npz_filepath, videos=self.videos, anomaly_labels=self.pixel_labels)

    @property
    def npz_filepath(self) -> str:
        return self.dataset_path + ".npz"

    # region Static loading methods
    @staticmethod
    def _get_videos_dictionary(dataset_path: str, extension: str):
        if not extension.startswith("."):
            extension = "." + extension
        sub_directories = [dir_info[0] for dir_info in os.walk(dataset_path)]
        if dataset_path in sub_directories:
            sub_directories.remove(dataset_path)
        tiff_images_dictionary = {}
        for sub_directory in sub_directories:
            tiff_images_paths = UCSDDataset._get_files_with_extension(sub_directory, extension)
            if len(tiff_images_paths) is not 0:
                tiff_images_dictionary[sub_directory] = tiff_images_paths
        return tiff_images_dictionary

    @staticmethod
    def _get_files_with_extension(directory: str, extension: str, sort=True):
        files = os.listdir(directory)
        result = []
        for file in files:
            if file.endswith(extension):
                result += [file]
        if sort:
            result.sort()
        return result

    @staticmethod
    def _load_videos_from_dictionary_of_paths(dictionary, dtype: Type = np.float32):
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

    @staticmethod
    def _load_ucsd_videos_raw(dataset_path: str):
        videos_dictionary = UCSDDataset._get_videos_dictionary(dataset_path, ".tif")
        return UCSDDataset._load_videos_from_dictionary_of_paths(videos_dictionary)
    # endregion
    # endregion

    @property
    def has_pixel_labels(self):
        return True
