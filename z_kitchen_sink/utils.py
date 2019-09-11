import tensorflow as tf
import numpy as np

from datasets import DatasetConfig, DatasetLoader
from models.AEP import get_temporal_loss_weights


class TmpModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath: str, verbose=0):
        super(TmpModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        target_path = self.filepath.format(epoch=epoch + 1)
        if self.verbose > 0:
            print("\nEpoch {epoch:05d}: saving model to {path}".format(epoch=epoch + 1, path=target_path))
        self.model.save(target_path,
                        include_optimizer=not tf.executing_eagerly())


def get_autoencoder_loss(input_length):
    temporal_loss_weights = get_temporal_loss_weights(input_length, input_length)

    def autoencoder_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        axis = list(range(2, y_true.shape.ndims))
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=axis)
        reconstruction_loss *= temporal_loss_weights
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    return autoencoder_loss


def eager_to_numpy(data):
    if isinstance(data, tuple):
        return tuple(eager_to_numpy(x) for x in data)
    elif isinstance(data, list):
        return [eager_to_numpy(x) for x in data]
    elif isinstance(data, np.ndarray):
        return data
    else:
        return data.numpy()


def get_landmarks_datasets():
    dataset_config = DatasetConfig("../datasets/emoly",
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    train_subset = dataset_loader.train_subset
    train_subset.subset_folders = [folder for folder in train_subset.subset_folders if "normal" in folder]

    test_subset = dataset_loader.test_subset
    test_subset.subset_folders = [folder for folder in test_subset.subset_folders if "induced" in folder]

    return dataset_loader, train_subset, test_subset
