import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, InputLayer, Dense, Layer, Reshape, Flatten
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Optional, Tuple

from datasets import DatasetLoader, DatasetConfig
from modalities import MelSpectrogram, ModalityShape
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class AnomalyTestLayer(Layer):
    def __init__(self,
                 score_threshold: float,
                 trainable=True,
                 name: str = None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super(AnomalyTestLayer, self).__init__(trainable=trainable,
                                               name=name,
                                               dtype=dtype,
                                               dynamic=dynamic,
                                               **kwargs)
        self.score_threshold = score_threshold

    def call(self, inputs, **kwargs):
        original, decoded = inputs
        reduction_axis = tuple(range(1, len(original.shape)))
        error = tf.reduce_mean(tf.square(original - decoded), axis=reduction_axis)
        return tf.sigmoid(error - self.score_threshold)

    def compute_output_shape(self, input_shape):
        return input_shape[:1]


class NPProp(object):
    def __init__(self, embeddings_size=40, mixture_count=16, gmm_updates_per_cycle=30):
        self.embeddings_size = embeddings_size

        self.encoder: Optional[Sequential] = None
        self.normal_decoder: Optional[Sequential] = None
        self.anomalous_decoder: Optional[Sequential] = None

        self.anomalous_autoencoder: Optional[Model] = None
        self.anomaly_detector: Optional[Model] = None

        self.gmm = GaussianMixture(n_components=mixture_count, max_iter=gmm_updates_per_cycle)
        # TODO : Fit GMM on Encoded normal sounds
        self.gmm.fit(np.random.normal(size=[200, embeddings_size], loc=5))
        self.z_threshold = -200.0

    # region Build models
    def build(self, input_shape: Tuple[int, ...]):
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

        self.encoder = Sequential(
            layers=
            [
                InputLayer(input_shape),
                Flatten(),
                Dense(units=512, activation="relu"),
                Dense(units=512, activation="relu"),
                Dense(units=self.embeddings_size),
            ],
            name="encoder")

        self.normal_decoder = Sequential(
            layers=
            [
                Dense(units=512, activation="relu"),
                Dense(units=512, activation="relu"),
                Dense(units=input_dim),
                Reshape(input_shape)
            ],
            name="normal_decoder"
        )

        self.anomalous_decoder = Sequential(
            layers=
            [
                Dense(units=512, activation="relu"),
                Dense(units=512, activation="relu"),
                Dense(units=input_dim),
                Reshape(input_shape)
            ],
            name="anomalous_decoder"
        )

        encoded = self.encoder(self.encoder.inputs)
        self.anomalous_autoencoder = Model(inputs=self.encoder.input,
                                           outputs=self.anomalous_decoder(encoded),
                                           name="anomalous_autoencoder")

        def anomalous_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return self.divergence(encoded) + reconstruction_loss

        self.anomalous_autoencoder.compile(optimizer="adam", loss=anomalous_loss)

        normal_decoded = self.normal_decoder(encoded)
        true_outputs_input_layer = Input(input_shape)
        detector_output = AnomalyTestLayer(0.05)([true_outputs_input_layer, normal_decoded])
        anomaly_detector_inputs = [self.encoder.input, true_outputs_input_layer]
        self.anomaly_detector = Model(inputs=anomaly_detector_inputs,
                                      outputs=detector_output,
                                      name="anomaly_detector")

        def detector_loss(y_true, y_pred):
            # y_true : 0 (normal) | 1 (anomaly)
            y_true = tf.reshape(y_true, tf.shape(y_pred))
            multipliers = tf.cast(y_true, tf.float32)
            # multipliers : 1 (normal) | -1 (anomaly)
            multipliers = (multipliers * - 2) + 1
            # Goal : Increase if anomaly (TPR) and lower if normal (FPR)
            return y_pred * multipliers

        self.anomaly_detector.compile(optimizer="adam", loss=detector_loss)

    @staticmethod
    def divergence(encoded):
        mean = tf.reduce_mean(encoded, axis=0)
        variance = tf.math.reduce_variance(encoded, axis=0)
        result = kullback_leibler_divergence_mean0_var1(mean, variance, use_variance_log=False)
        return result

    # endregion

    # region Train
    def train(self,
              dataset_loader: DatasetLoader,
              batch_size: int,
              cycles: int,
              epochs_per_cycle: int,
              steps_per_epoch: int):

        anomalous_dataset, detection_dataset = self.prepare_datasets(dataset_loader, batch_size)

        for cycle_index in range(cycles):
            self.update_anomalous_autoencoder(anomalous_dataset, epochs_per_cycle, steps_per_epoch)
            self.update_anomaly_detector(detection_dataset, epochs_per_cycle, steps_per_epoch)
            self.update_gmm()

    # region Datasets
    def prepare_datasets(self,
                         dataset_loader: DatasetLoader,
                         batch_size: int):
        anomalous_dataset = self.prepare_anomalous_dataset(dataset_loader, batch_size)
        detection_dataset = self.prepare_detection_dataset(dataset_loader, batch_size)
        return anomalous_dataset, detection_dataset

    @staticmethod
    def prepare_anomalous_dataset(dataset_loader: DatasetLoader,
                                  batch_size: int):
        subset = dataset_loader.train_subset
        anomalous_samples = [folder for folder in subset.subset_folders if "normal" not in folder]
        dataset = subset.make_tf_dataset(output_labels=False, subset_folder=anomalous_samples)
        dataset = dataset.batch(batch_size).prefetch(-1)
        return dataset

    def prepare_detection_dataset(self,
                                  dataset_loader: DatasetLoader,
                                  batch_size: int):
        if (batch_size % 2) != 0:
            raise ValueError("Batch size must be even")

        subset = dataset_loader.train_subset
        normal_samples = [folder for folder in subset.subset_folders if "normal" in folder]
        normal_dataset = subset.make_tf_dataset(output_labels=False, subset_folder=normal_samples)
        normal_dataset = normal_dataset.batch(batch_size // 2)

        anomalous_dataset = tf.data.Dataset.from_generator(self.rejection_sampling,
                                                           output_types=tf.float32,
                                                           output_shapes=[self.embeddings_size])

        anomalous_dataset = anomalous_dataset.batch(batch_size // 2)
        anomalous_dataset = anomalous_dataset.map(self.anomalous_decoder)
        anomalous_dataset = anomalous_dataset.map(lambda x: (x, x))

        datasets = [normal_dataset, anomalous_dataset]
        choice_dataset = tf.data.Dataset.range(2).repeat(-1)
        detection_dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

        detection_dataset = detection_dataset.apply(tf.data.experimental.unbatch())
        detection_dataset = detection_dataset.batch(batch_size)
        detection_dataset = detection_dataset.map(self.label_detection_dataset)

        return detection_dataset

    def rejection_sampling(self):
        while True:
            z = np.random.normal(size=[self.embeddings_size])
            z_log_likelihood = self.gmm.score([z])
            if z_log_likelihood < self.z_threshold:
                yield z

    @staticmethod
    def label_detection_dataset(x, y):
        half_batch_size = tf.shape(x)[0] / 2
        normal_labels = tf.zeros([half_batch_size], dtype=tf.int32)
        anomalous_labels = tf.ones([half_batch_size], dtype=tf.int32)
        labels = tf.concat([normal_labels, anomalous_labels], axis=0, name="concat_detection_labels")
        return (x, y), labels

    # endregion

    # region Updates (Anomalous AE -> Anomaly Detector -> GMM)
    def update_anomalous_autoencoder(self,
                                     dataset: tf.data.Dataset,
                                     epochs: int,
                                     steps_per_epoch: int):
        self.anomalous_autoencoder.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def update_anomaly_detector(self,
                                detection_dataset: tf.data.Dataset,
                                epochs: int,
                                steps_per_epoch: int):
        self.anomaly_detector.fit(detection_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def update_gmm(self):
        pass
    # endregion
    # endregion


def main():
    np_prop = NPProp()
    np_prop.build(input_shape=(11, 40))

    dataset_config = DatasetConfig("../datasets/emoly",
                                   modalities_io_shapes=
                                   {
                                       MelSpectrogram: ModalityShape((11, 40),
                                                                     (11, 40))
                                   },
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(dataset_config)

    np_prop.train(dataset_loader=dataset_loader,
                  batch_size=16,
                  cycles=20,
                  epochs_per_cycle=2,
                  steps_per_epoch=100)


if __name__ == "__main__":
    main()
