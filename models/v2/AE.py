import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import os
import time
from typing import Union, List

from datasets import DatasetLoader, DatasetConfig, SubsetLoader
from modalities import RawVideo, ModalityLoadInfo
from callbacks import AUCCallback
from layers.utility_layers import RawPredictionsLayer


class AE(keras.Model):
    def __init__(self,
                 encoder: Union[keras.Model, keras.Sequential, List[keras.layers.Layer]],
                 decoder: Union[keras.Model, keras.Sequential, List[keras.layers.Layer]],
                 **kwargs):
        if "optimizer" in kwargs:
            self.optimizer: keras.optimizers.Optimizer = kwargs.pop("optimizer")

        super(AE, self).__init__(**kwargs)
        self.__dict__.update(kwargs)

        if isinstance(encoder, list):
            encoder = keras.Sequential(encoder, name="encoder")
        if isinstance(decoder, list):
            decoder = keras.Sequential(decoder, name="decoder")

        self.encoder: keras.Model = encoder
        self.decoder: keras.Model = decoder

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def forward(self, x):
        return self.decode(self.encode(x))

    @tf.function
    def compute_loss(self, x):
        decoded = self.forward(x)
        reconstruction_error = tf.reduce_mean(tf.square(decoded - x))
        return reconstruction_error

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def train_step(self, x):
        gradients, loss = self.compute_gradients(x)
        grads_and_vars = zip(gradients, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss


def main():
    input_channels = 1
    input_shape = (16, 128, 128, input_channels)
    epoch_length = 2
    batch_size = 8

    # region Model
    encoder = \
        [
            keras.layers.InputLayer(input_shape),
            keras.layers.Conv3D(filters=32, kernel_size=3, strides=(2, 2, 2), padding="same", activation="elu"),
            keras.layers.Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same", activation="elu"),
            keras.layers.Conv3D(filters=64, kernel_size=3, strides=(2, 2, 2), padding="same", activation="elu"),
            keras.layers.Conv3D(filters=64, kernel_size=3, strides=(1, 2, 2), padding="same", activation="elu"),
            keras.layers.Conv3D(filters=128, kernel_size=3, strides=(2, 2, 2), padding="same", activation="elu"),
            keras.layers.Conv3D(filters=128, kernel_size=3, strides=(1, 2, 2), padding="same", activation="elu"),
            keras.layers.Flatten(),
            keras.layers.Dense(units=256)
        ]

    decoder = \
        [
            keras.layers.Dense(units=2 * 2 * 2 * 128, activation="elu"),
            keras.layers.Reshape(target_shape=(2, 2, 2, 128)),
            keras.layers.Conv3DTranspose(filters=128, kernel_size=3, strides=(1, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=128, kernel_size=3, strides=(2, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=64, kernel_size=3, strides=(1, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=64, kernel_size=3, strides=(2, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=32, kernel_size=3, strides=(2, 2, 2),
                                         padding="same", activation="elu"),
            keras.layers.Conv3DTranspose(filters=input_channels, kernel_size=3, strides=1,
                                         padding="same", activation="sigmoid"),
        ]

    optimizer = keras.optimizers.Adam(lr=2e-4)

    model = AE(encoder, decoder, optimizer=optimizer)
    # endregion

    # region Dataset

    modalities_pattern = (
        ModalityLoadInfo(RawVideo, input_shape[0], input_shape),
        ModalityLoadInfo(RawVideo, input_shape[0], input_shape)
    )
    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           output_range=(0.0, 1.0))

    dataset_loader = DatasetLoader(config)
    train_dataset = dataset_loader.train_subset.make_tf_dataset(modalities_pattern)
    train_dataset = train_dataset.map(lambda x, y: x)
    train_dataset = train_dataset.batch(batch_size)
    # endregion

    log_dir = "../logs/tests/ae/log_{}".format(int(time.time()))
    os.makedirs(log_dir)
    tensorboard = keras.callbacks.TensorBoard(log_dir)

    decoded = model.decoder(model.encoder(model.encoder.input))
    raw_predictions = RawPredictionsLayer()([decoded, model.encoder.input])
    raw_predictions_model = keras.Model(inputs=model.encoder.input, outputs=raw_predictions,
                                        name="raw_predictions_model")

    anomaly_modalities_pattern = (
        *modalities_pattern,
        "labels"
    )
    inputs, outputs, labels = dataset_loader.test_subset.get_batch(batch_size=16,
                                                                   pattern=anomaly_modalities_pattern)
    labels = SubsetLoader.timestamps_labels_to_frame_labels(labels.numpy(), frame_count=16)

    auc_callback = AUCCallback(predictions_model=raw_predictions_model,
                               tensorboard=tensorboard,
                               inputs=inputs,
                               outputs=outputs,
                               labels=labels,
                               epoch_freq=1,
                               plot_size=(128, 128)
                               )

    callbacks = [tensorboard, auc_callback]

    for epoch in range(50):
        losses = []
        for batch_index, batch in tqdm(zip(range(epoch_length), train_dataset), total=epoch_length):
            loss = model.train_step(batch)
            losses.append(loss.numpy())
            print("Epoch: {} | MSE: {}".format(epoch, np.mean(losses)))

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={"loss": np.mean(losses)})


if __name__ == "__main__":
    print(tf.__version__)

    pass
