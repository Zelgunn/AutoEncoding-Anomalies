from keras.utils import Sequence
import numpy as np


class NoisyImagesGenerator(Sequence):
    def __init__(self,
                 images: np.ndarray,
                 dropout_rate: float,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle=True):
        self.images = images
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_length = epoch_length

        if self.shuffle:
            np.random.shuffle(self.images)

        self.index = 0

    def __getitem__(self, item):
        if self.epoch_length is None:
            images: np.ndarray = self.images[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        else:
            indices = np.random.permutation(np.arange(self.images.shape[0]))[:self.batch_size]
            images: np.ndarray = self.images[indices]

        noisy_images = images
        if self.dropout_rate > 0.0:
            noisy_images = np.copy(noisy_images)
            noisy_images = np.reshape(noisy_images, [images.shape[0], -1])
            dropout_count = int(np.ceil(self.dropout_rate * noisy_images.shape[1]))
            for i in range(images.shape[0]):
                dropout_indices = np.random.permutation(np.arange(noisy_images.shape[1]))[:dropout_count]
                noisy_images[i][dropout_indices] = 0.0
            noisy_images = np.reshape(noisy_images, images.shape)

        self.index += 1
        return noisy_images, images

    def __len__(self):
        if self.epoch_length is None:
            return int(np.ceil(self.images.shape[0]) / self.batch_size)
        else:
            return self.epoch_length

    def on_epoch_end(self):
        if self.epoch_length is None:
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.images)
