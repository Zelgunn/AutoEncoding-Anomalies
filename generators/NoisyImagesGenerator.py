import numpy as np

from scheme import DataGenerator


class NoisyImagesGenerator(DataGenerator):
    def __init__(self,
                 images: np.ndarray,
                 dropout_rate: float,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle=True):
        super(NoisyImagesGenerator, self).__init__(batch_size, epoch_length, shuffle)
        self.images = images
        self.dropout_rate = dropout_rate

        if self.shuffle_on_epoch_end:
            self.shuffle()

        self.index = 0

    def add_noise_to(self, images, dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        noisy_images = images
        if dropout_rate > 0.0:
            noisy_images = np.copy(noisy_images)
            noisy_images = np.reshape(noisy_images, [images.shape[0], -1])
            dropout_count = int(np.ceil(dropout_rate * noisy_images.shape[1]))
            for i in range(images.shape[0]):
                dropout_indices = np.random.permutation(np.arange(noisy_images.shape[1]))[:dropout_count]
                noisy_images[i][dropout_indices] = 0.0
            noisy_images = np.reshape(noisy_images, images.shape)
        return noisy_images

    @property
    def samples_count(self):
        return self.images.shape[0]

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        indices = np.random.permutation(np.arange(self.images.shape[0]))[:batch_size]
        images: np.ndarray = self.images[indices]
        noisy_images = self.add_noise_to(images)

        return noisy_images, images

    def current_batch(self):
        images: np.ndarray = self.images[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        noisy_images = self.add_noise_to(images)
        return noisy_images, images

    def shuffle(self):
        np.random.shuffle(self.images)
