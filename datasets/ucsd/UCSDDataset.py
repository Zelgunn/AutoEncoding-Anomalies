import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Type

from scheme import Dataset


class UCSDDataset(Dataset):
    def load(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path
        if self.dataset_path.endswith(os.path.sep):
            self.dataset_path = self.dataset_path[:-1]

        self.anomaly_labels = None

        self.saved_to_npz = os.path.exists(self.npz_filepath)
        print("Loading UCSD dataset : " + dataset_path)
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
            self.images = UCSDDataset._load_ucsd_images_raw(dataset_path)
            bmp_images_dictionary = UCSDDataset._get_images_dictionary(dataset_path, ".bmp")
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

    def next_batch(self, batch_size: int, complete_batch_at_end=True):
        if self.epochs_completed == 0 and self.index_in_epoch == 0:
            self.shuffle()

        if (self.index_in_epoch + batch_size) > self.samples_count:
            self.epochs_completed += 1
            remaining_images = self.images[self.index_in_epoch: self.samples_count]
            self.shuffle()
            if complete_batch_at_end:
                self.index_in_epoch = batch_size - (self.samples_count - self.index_in_epoch)
                images = np.concatenate([remaining_images, self.images[0: self.index_in_epoch]], axis=0)
                return images
            else:
                self.index_in_epoch = 0
                return remaining_images

        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        return self.images[start: self.index_in_epoch]

    def normalize(self, current_min, current_max, target_min=0.0, target_max=1.0):
        self.images = (self.images - current_min) / (current_max - current_min) * (target_max - target_min) + target_min

    @property
    def npz_filepath(self) -> str:
        return self.dataset_path + ".npz"

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
    def _load_ucsd_images(dataset_path: str):
        images_npz_path = os.path.join(dataset_path + ".npz")
        if os.path.exists(images_npz_path):
            images = np.load(images_npz_path)
            return images
        else:
            return UCSDDataset._load_ucsd_images_raw(dataset_path)

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
