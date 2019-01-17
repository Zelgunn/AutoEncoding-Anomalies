from abc import ABC, abstractmethod
import numpy as np
import cv2


class Dataset(ABC):
    def __init__(self, **kwargs):
        self.images: np.ndarray = None
        self.anomaly_labels = None

        self.dataset_path = None
        self.saved_to_npz = False
        self.index_in_epoch = 0
        self.epochs_completed = 0

        if "images" in kwargs:
            self.images = kwargs["images"]

        if "anomaly_labels" in kwargs:
            self.anomaly_labels = kwargs["anomaly_labels"]

        if "dataset_path" in kwargs:
            self.dataset_path = kwargs["dataset_path"]
            if self.images is None:
                self.load(self.dataset_path)

    @abstractmethod
    def load(self, dataset_path: str, **kwargs):
        raise NotImplementedError

    def shuffle(self):
        if self.anomaly_labels is None:
            np.random.shuffle(self.images)
        else:
            shuffle_indices = np.random.permutation(np.arange(self.samples_count))
            self.images = self.images[shuffle_indices]
            self.anomaly_labels = self.anomaly_labels[shuffle_indices]

    def resized(self, size):
        images = resize_images(self.images, size)
        if self.anomaly_labels is None:
            anomaly_labels = None
        else:
            anomaly_labels = resize_images(self.anomaly_labels.astype(np.float32), size).astype(bool)

        dataset_type = type(self)
        dataset: Dataset = dataset_type(images=images, anomaly_labels=anomaly_labels)
        return dataset

    @property
    def samples_count(self):
        assert self.images is not None
        return self.images.shape[0]

    @property
    def images_size(self):
        assert self.images is not None
        return self.images.shape[1:3]

    def frame_level_labels(self):
        result = np.any(self.anomaly_labels, axis=(1, 2, 3))
        return result


def resize_images(images, size) -> np.ndarray:
    result = np.empty((images.shape[0], *size, images.shape[3]))
    for i in range(images.shape[0]):
        resized_one = cv2.resize(images[i], tuple(reversed(size)))
        if images.shape[3] == 1:
            result[i, :, :, 0] = resized_one
        else:
            result[i, :, :] = resized_one
    return result