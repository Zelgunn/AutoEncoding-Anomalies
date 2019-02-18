from abc import ABC, abstractmethod
import tensorflow as tf

from keras.callbacks import Callback, TensorBoard


class TensorBoardPlugin(Callback, ABC):
    def __init__(self,
                 tensorboard: TensorBoard,
                 update_freq: str or int,
                 epoch_freq: int = None):
        super(TensorBoardPlugin, self).__init__()
        self.tensorboard = tensorboard

        self.update_freq: int or str = None
        if update_freq == "batch":
            self.update_freq = 1
        else:
            self.update_freq = update_freq

        if update_freq == "epoch":
            self.epoch_freq = epoch_freq
        else:
            self.epoch_freq = None

        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != "epoch":
            self.samples_seen += logs["size"]
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen

    def on_epoch_end(self, epoch, logs=None):
        if self.update_freq == "epoch":
            index = epoch
        else:
            index = self.samples_seen

        if (self.epoch_freq is None) or (((epoch + 1) % self.epoch_freq) == 0):
            self._write_logs(index)

    @property
    def session(self) -> tf.Session:
        return self.tensorboard.sess

    @property
    def writer(self) -> tf.summary.FileWriter:
        return self.tensorboard.writer

    @abstractmethod
    def _write_logs(self, index):
        raise NotImplementedError
