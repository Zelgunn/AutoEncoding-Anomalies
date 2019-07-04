import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, InputLayer, Dense, Layer, Reshape, Flatten, Lambda
from tensorflow.python.keras.optimizers import Adam
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Optional, Tuple

from datasets import DatasetLoader, DatasetConfig, SubsetLoader
from modalities import MelSpectrogram, ModalityShape
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class AnomalyTestLayer(Layer):
    def __init__(self,
                 desired_false_positive_rate: float,
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
        self.desired_false_positive_rate = desired_false_positive_rate
        self.top_k_encoded: Optional[tf.Variable] = None

    def build(self, input_shape):
        code_size = input_shape[-1][-1]
        self.top_k_encoded = tf.Variable(initial_value=np.random.normal(size=[code_size]),
                                         dtype=tf.float32,
                                         name="top_k_encoded",
                                         trainable=False)

    def call(self, inputs, **kwargs):
        # TODO : Need Encoded here to compute z_threshold
        original, decoded, encoded = inputs

        reduction_axis = tuple(range(1, len(original.shape)))
        error = tf.reduce_mean(tf.square(original - decoded), axis=reduction_axis)

        half_batch_size = tf.shape(error)[0] // 2
        k = tf.floor(self.desired_false_positive_rate * tf.cast(half_batch_size, tf.float32))
        k = tf.cast(k, tf.int32)
        top_k_error, top_k_error_indexes = tf.math.top_k(error[:half_batch_size], k=k)
        numeric_threshold = top_k_error[-1]

        assign_top_k_encoded = self.top_k_encoded.assign(encoded[top_k_error_indexes[-1]])
        with tf.control_dependencies([assign_top_k_encoded]):
            test_results = tf.sigmoid(error - numeric_threshold)

        return test_results

    def compute_output_shape(self, input_shape):
        return input_shape[0][:1]


class DetectorInputLayer(Layer):
    def __init__(self,
                 anomalous_decoder: Model,
                 trainable=True,
                 name: str = None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super(DetectorInputLayer, self).__init__(trainable=trainable,
                                                 name=name,
                                                 dtype=dtype,
                                                 dynamic=dynamic,
                                                 **kwargs)
        self.anomalous_decoder = anomalous_decoder

    def call(self, inputs, **kwargs):
        anomalous_code, normal_inputs, normal_outputs = inputs
        generated_anomalies = self.anomalous_decoder(anomalous_code)
        inputs = tf.concat([normal_inputs, generated_anomalies], axis=0)
        outputs = tf.concat([normal_outputs, generated_anomalies], axis=0)
        return inputs, outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1:]


class NPProp(object):
    def __init__(self, embeddings_size=40, mixture_count=16, gmm_updates_per_cycle=30):
        self.embeddings_size = embeddings_size
        self.mixture_count = mixture_count
        self.gmm_updates_per_cycle = gmm_updates_per_cycle

        self.anomaly_test_layer: Optional[AnomalyTestLayer] = None

        self.encoder: Optional[Sequential] = None
        self.normal_decoder: Optional[Sequential] = None
        self.anomalous_decoder: Optional[Sequential] = None

        self.anomalous_autoencoder: Optional[Model] = None
        self.anomaly_detector: Optional[Model] = None
        self.anomaly_detector_trainer: Optional[Model] = None

        self.gmm: Optional[GaussianMixture] = None
        # TODO : Compute z_threshold
        self.z_threshold = -200.0

    # region Build models
    def build(self, input_shape: Tuple[int, ...]):
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim

        self.encoder = Sequential(
            layers=
            [
                InputLayer(input_shape, name="encoder_input_layer"),
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
                InputLayer([self.embeddings_size], name="anomalous_decoder_input_layer"),
                Dense(units=512, activation="relu"),
                Dense(units=512, activation="relu"),
                Dense(units=input_dim),
                Reshape(input_shape)
            ],
            name="anomalous_decoder"
        )

        self.anomaly_test_layer = AnomalyTestLayer(0.2)

        encoded = self.encoder(self.encoder.inputs)
        self.anomalous_autoencoder = Model(inputs=self.encoder.input,
                                           outputs=self.anomalous_decoder(encoded),
                                           name="anomalous_autoencoder")

        def anomalous_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return self.divergence(encoded) + reconstruction_loss

        self.anomalous_autoencoder.compile(optimizer=Adam(lr=1e-4), loss=anomalous_loss)

        # region Anomaly detector
        normal_decoded = self.normal_decoder(encoded)
        true_outputs_input_layer = Input(input_shape, name="detector_true_outputs")
        detector_output = self.anomaly_test_layer([true_outputs_input_layer, normal_decoded, encoded])
        anomaly_detector_inputs = [self.encoder.input, true_outputs_input_layer]
        self.anomaly_detector = Model(inputs=anomaly_detector_inputs,
                                      outputs=detector_output,
                                      name="anomaly_detector")

        # endregion

        # region Anomaly detector trainer
        def detector_loss(y_true, y_pred):
            # y_true : 0 (normal) | 1 (anomaly)
            y_true = tf.reshape(y_true, tf.shape(y_pred))
            # multipliers : 1 (normal) | -1 (anomaly)
            multipliers = (y_true * - 2) + 1
            # Goal : Increase if anomaly (TPR) and lower if normal (FPR)
            return y_pred * multipliers

        anomalous_code_input_layer = Input(shape=[self.embeddings_size], name="anomalous_code")
        normal_inputs_input_layer = Input(shape=input_shape, name="normal_inputs")
        detector_inputs = [anomalous_code_input_layer, normal_inputs_input_layer, true_outputs_input_layer]
        non_trainable_anomalous_decoder = Model(inputs=self.anomalous_decoder.input,
                                                outputs=self.anomalous_decoder.output,
                                                name="non_trainable_anomalous_decoder",
                                                trainable=False)
        detector_input_layer = DetectorInputLayer(non_trainable_anomalous_decoder)(detector_inputs)
        detector_output_layer = self.anomaly_detector(detector_input_layer)
        self.anomaly_detector_trainer = Model(inputs=detector_inputs,
                                              outputs=detector_output_layer,
                                              name="anomaly_detector_trainer")
        self.anomaly_detector_trainer.compile(optimizer=Adam(lr=1e-4), loss=detector_loss)
        # endregion

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
              epochs: int,
              steps_per_epoch: int,
              gmm_sample_count: int):

        anomalous_dataset, detection_dataset, normal_code_dataset = self.prepare_datasets(dataset_loader, batch_size)
        gmm_batch_count = int(np.ceil(gmm_sample_count / batch_size))

        if self.gmm is None:
            print("Initializing GMM")
            self.gmm = GaussianMixture(n_components=self.mixture_count, max_iter=self.gmm_updates_per_cycle)
            self.update_gmm(normal_code_dataset, gmm_batch_count)

        for cycle_index in range(epochs):
            print("Fitting Anomalous autoencoder")
            self.update_anomalous_autoencoder(anomalous_dataset, steps_per_epoch)
            print("Fitting Anomaly detector")
            self.update_anomaly_detector(detection_dataset, steps_per_epoch)
            self.z_threshold = self.gmm.score([self.anomaly_test_layer.top_k_encoded.numpy()])
            if self.z_threshold < - 500:
                self.z_threshold = -500
            print("New z threshold:", self.z_threshold)
            print("Fitting GMM")
            self.update_gmm(normal_code_dataset, gmm_batch_count)

    # region Datasets
    def prepare_datasets(self,
                         dataset_loader: DatasetLoader,
                         batch_size: int
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        anomalous_dataset = self.prepare_anomalous_dataset(dataset_loader, batch_size)
        normal_dataset = self.prepare_base_normal_dataset(dataset_loader)
        detection_dataset = self.prepare_detection_dataset(normal_dataset, batch_size)
        normal_code_dataset = self.prepare_normal_code_dataset(normal_dataset, batch_size)
        return anomalous_dataset, detection_dataset, normal_code_dataset

    @staticmethod
    def prepare_anomalous_dataset(dataset_loader: DatasetLoader,
                                  batch_size: int
                                  ) -> tf.data.Dataset:
        subset = dataset_loader.train_subset
        anomalous_samples = [folder for folder in subset.subset_folders if "normal" not in folder]
        dataset = subset.make_tf_dataset(output_labels=False, subset_folder=anomalous_samples)
        dataset = dataset.batch(batch_size).prefetch(-1)
        return dataset

    @staticmethod
    def prepare_base_normal_dataset(dataset_loader: DatasetLoader):
        subset = dataset_loader.train_subset
        normal_samples = [folder for folder in subset.subset_folders if "normal" in folder]
        dataset = subset.make_tf_dataset(output_labels=False, subset_folder=normal_samples)
        return dataset

    def prepare_detection_dataset(self,
                                  normal_dataset: tf.data.Dataset,
                                  batch_size: int
                                  ) -> tf.data.Dataset:
        if (batch_size % 2) != 0:
            raise ValueError("Batch size must be even")

        normal_dataset = normal_dataset.batch(batch_size // 2)

        anomalous_dataset = tf.data.Dataset.from_generator(self.rejection_sampling,
                                                           output_types=tf.float32,
                                                           output_shapes=[self.embeddings_size])
        anomalous_dataset = anomalous_dataset.batch(batch_size // 2)

        detection_dataset = tf.data.Dataset.zip((anomalous_dataset, normal_dataset))
        detection_dataset = detection_dataset.map(self.label_detection_dataset)

        return detection_dataset

    @staticmethod
    def prepare_normal_code_dataset(normal_dataset: tf.data.Dataset,
                                    batch_size: int
                                    ) -> tf.data.Dataset:
        dataset = normal_dataset.map(lambda x, y: x)
        dataset = dataset.batch(batch_size)
        return dataset

    def rejection_sampling(self):
        while True:
            z = np.random.normal(size=[self.embeddings_size])
            z_log_likelihood = self.gmm.score([z])
            if z_log_likelihood < self.z_threshold:
                yield z

    @staticmethod
    def label_detection_dataset(anomalous_code, normal_io):
        normal_inputs, normal_outputs = normal_io

        half_batch_size = tf.shape(normal_inputs)[0]
        normal_labels = tf.zeros([half_batch_size], dtype=tf.float32)
        anomalous_labels = tf.ones([half_batch_size], dtype=tf.float32)
        labels = tf.concat([normal_labels, anomalous_labels], axis=0, name="concat_detection_labels")
        return (anomalous_code, normal_inputs, normal_outputs), labels

    # endregion

    # region Updates (Anomalous AE -> Anomaly Detector -> GMM)
    def update_anomalous_autoencoder(self,
                                     anomalous_dataset: tf.data.Dataset,
                                     steps_per_epoch: int):
        self.anomalous_autoencoder.fit(anomalous_dataset, epochs=1, steps_per_epoch=steps_per_epoch)

    def update_anomaly_detector(self,
                                detection_dataset: tf.data.Dataset,
                                steps_per_epoch: int):
        self.anomaly_detector_trainer.fit(detection_dataset, epochs=1, steps_per_epoch=steps_per_epoch)

    def update_gmm(self,
                   normal_code_dataset: tf.data.Dataset,
                   gmm_batch_count: int
                   ):
        samples = self.encoder.predict(normal_code_dataset, steps=gmm_batch_count)
        self.gmm.fit(samples)

    # endregion
    # endregion

    def evaluate(self,
                 dataset_loader: DatasetLoader,
                 batch_size: int):
        subset = dataset_loader.test_subset
        dataset = subset.labeled_tf_dataset.batch(batch_size).prefetch(-1)
        dataset = dataset.map(lambda *x: (x,))
        labels_input_layer = Input(shape=[None, 2], name="labels_input_layer")
        labels_output_layer = Lambda(function=tf.identity)(labels_input_layer)
        detector_wrapper = Model(inputs=[*self.anomaly_detector.inputs, labels_input_layer],
                                 outputs=[*self.anomaly_detector.outputs, labels_output_layer],
                                 name="anomaly_detector_wrapper")
        predictions, ground_truth = detector_wrapper.predict(dataset, steps=1000)
        ground_truth = SubsetLoader.timestamps_labels_to_frame_labels(ground_truth, 11)
        ground_truth = np.any(ground_truth, axis=-1)

        roc = tf.metrics.AUC()
        roc.update_state(ground_truth, predictions)
        print("AUC : ", roc.result().numpy())


def main():
    sequence_length = 11
    mel_filter_count = 100

    np_prop = NPProp(gmm_updates_per_cycle=30)
    np_prop.build(input_shape=(sequence_length, mel_filter_count))

    dataset_config = DatasetConfig("E:/datasets/emoly",
                                   modalities_pattern=
                                   {
                                       MelSpectrogram: ModalityShape((sequence_length, mel_filter_count),
                                                                     (sequence_length, mel_filter_count))
                                   },
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(dataset_config)

    try:
        np_prop.train(dataset_loader=dataset_loader,
                      batch_size=512,
                      epochs=2000,
                      steps_per_epoch=50,
                      gmm_sample_count=512 * 10)
    except KeyboardInterrupt:
        pass

    np_prop.evaluate(dataset_loader=dataset_loader,
                     batch_size=128)


if __name__ == "__main__":
    main()
