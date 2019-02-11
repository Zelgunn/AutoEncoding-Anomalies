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

        self.anomaly_labels = None

        self.saved_to_npz = os.path.exists(self.npz_filepath)
        print("Loading UCSD dataset : " + self.dataset_path)
        if self.saved_to_npz:
            print("Found saved dataset (in .npz file)")
            npz_file = np.load(self.npz_filepath)
            self.images = npz_file["images"]
            if "anomaly_labels" in npz_file:
                self.anomaly_labels = npz_file["anomaly_labels"]
                if None in self.anomaly_labels:
                    self.anomaly_labels = None
        else:
            print("Building dataset from raw data")
            self.images = UCSDDataset._load_ucsd_images_raw(self.dataset_path)
            bmp_images_dictionary = UCSDDataset._get_images_dictionary(self.dataset_path, ".bmp")
            if len(bmp_images_dictionary) > 0:
                self.anomaly_labels = UCSDDataset._load_images_from_dictionary_of_paths(bmp_images_dictionary,
                                                                                        dtype=np.bool)
            else:
                self.anomaly_labels = None

    def save_to_npz(self, force=False):
        if self.saved_to_npz and not force:
            return
        print("Saving UCSD dataset : " + self.dataset_path + " (to .npz file)")
        np.savez_compressed(self.npz_filepath, images=self.images, anomaly_labels=self.anomaly_labels)

    @property
    def npz_filepath(self) -> str:
        return self.dataset_path + ".npz"

    # region Static loading methods
    @staticmethod
    def _get_images_dictionary(dataset_path: str, extension: str):
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
    def _load_images_from_dictionary_of_paths(dictionary, dtype: Type = np.float32):
        images_count = 0
        for _, tiff_images_paths in dictionary.items():
            images_count += len(tiff_images_paths)

        images = None
        index = 0
        sub_directories = dictionary.keys()
        sub_directories = sorted(sub_directories)
        for sub_directory in sub_directories:
            tiff_images_paths = dictionary[sub_directory]
            images_count_in_folder = len(tiff_images_paths)
            sub_directory_info: str = sub_directory.split(os.path.sep)[-1]
            for i in tqdm(range(images_count_in_folder), desc=sub_directory_info):
                image_path = os.path.join(sub_directory, tiff_images_paths[i])
                image = Image.open(image_path)
                image = np.array(image)
                if images is None:
                    images = np.ndarray([images_count, image.shape[0], image.shape[1]], dtype=dtype)
                images[index] = image
                index += 1
        images = np.expand_dims(images, axis=-1)
        return images

    @staticmethod
    def _load_ucsd_images_raw(dataset_path: str):
        tiff_images_dictionary = UCSDDataset._get_images_dictionary(dataset_path, ".tif")
        return UCSDDataset._load_images_from_dictionary_of_paths(tiff_images_dictionary)
    # endregion
    # endregion

    @property
    def has_pixel_level_anomaly_labels(self):
        return True
