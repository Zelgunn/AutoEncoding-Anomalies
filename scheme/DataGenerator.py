from keras.utils import Sequence
from abc import ABC, abstractmethod
import numpy as np


class DataGenerator(Sequence, ABC):
    def __init__(self,
                 batch_size=64,
                 epoch_length: int = None,
                 shuffle_on_epoch_end=True):
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.epoch_length = epoch_length
        self.index = 0

    def __getitem__(self, index):
        batch = self.current_batch() if self.epoch_length is None else self.sample()
        self.index += 1
        return batch

    def __len__(self):
        if self.epoch_length is None:
            return int(np.ceil(self.samples_count / self.batch_size))
        else:
            return self.epoch_length

    @property
    @abstractmethod
    def samples_count(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size=None):
        raise NotImplementedError

    @abstractmethod
    def current_batch(self):
        raise NotImplementedError

    def shuffle(self):
        pass

    def on_epoch_end(self):
        if self.epoch_length is None and self.shuffle_on_epoch_end:
            self.shuffle()
