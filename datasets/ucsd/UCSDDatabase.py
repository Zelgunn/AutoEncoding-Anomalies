from datasets.ucsd.UCSDDataset import UCSDDataset
import os
import numpy as np
import cv2

from scheme import Database


class UCSDDatabase(Database):
    def load(self, database_path, **kwargs):
        if database_path.endswith("\\"):
            database_path = database_path[:-1]
        self.database_path = database_path

        self.train_dataset = self.load_dataset("Train")
        self.test_dataset = self.load_dataset("Test")

        self.normalize()

        if not self.train_dataset.saved_to_npz:
            self.train_dataset.save_to_npz()
            self.test_dataset.save_to_npz()

    def get_images_shape(self):
        return self.train_dataset.images.shape[1:4]

    def shuffle(self, seed=None):
        np.random.seed(seed)
        self.train_dataset.shuffle()
        self.test_dataset.shuffle()

    def load_dataset(self, dataset_name: str):
        dataset_path = os.path.join(self.database_path, dataset_name)
        dataset = UCSDDataset(dataset_path=dataset_path)
        return dataset

    def normalize(self, target_min=0.0, target_max=1.0):
        current_min = min(self.train_dataset.images.min(), self.test_dataset.images.min())
        current_max = max(self.train_dataset.images.max(), self.test_dataset.images.max())
        self.train_dataset.normalize(current_min, current_max, target_min, target_max)
        self.test_dataset.normalize(current_min, current_max, target_min, target_max)

    def visualize_test_dataset(self):
        labels = self.test_dataset.anomaly_labels

        for i in range(len(labels)):
            tmp: np.ndarray = self.test_dataset.anomaly_labels[i]
            tmp = tmp.astype(np.float32)
            inv_tmp = 1.0 - tmp
            tmp = self.test_dataset.images[i] * tmp
            tmp += self.test_dataset.images[i] * inv_tmp * 0.25
            cv2.imshow("test_dataset", tmp)
            cv2.waitKey(30)
