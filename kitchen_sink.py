import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation, Lambda
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, Reshape, Dense
from tensorflow.python.keras.layers import CuDNNLSTM, RepeatVector
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, TerminateOnNaN
from tensorflow.python.keras.regularizers import l1
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from typing import Tuple, Type, Optional

from callbacks import ImageCallback, AUCCallback
from datasets.loaders import DatasetConfig, DatasetLoader, SubsetLoader
from modalities import ModalityShape, RawVideo, Modality
from layers.utility_layers import RawPredictionsLayer


# region Callbacks
def make_auc_callback(test_subset: SubsetLoader,
                      predictions_model: Model,
                      tensorboard: TensorBoard
                      ) -> AUCCallback:
    inputs, outputs, labels = test_subset.get_batch(batch_size=1024, output_labels=True)
    labels = SubsetLoader.timestamps_labels_to_frame_labels(labels.numpy(), inputs.shape[1])

    auc_callback = AUCCallback(predictions_model=predictions_model,
                               tensorboard=tensorboard,
                               inputs=inputs,
                               outputs=outputs,
                               labels=labels,
                               epoch_freq=1,
                               plot_size=(128, 128),
                               batch_size=128)
    return auc_callback


class TmpModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save("../logs/tests/kitchen_sink/mfcc_only/weights_{epoch:03d}.hdf5".format(epoch=epoch),
                        include_optimizer=False)


# endregion

# region Super Test
class AnomalyDetector(object):
    def __init__(self,
                 autoencoder: Model,
                 ):
        self.autoencoder: Model = autoencoder
        self._raw_predictions_model: Optional[Model] = None

    def make_raw_predictions_model(self, include_labels_io=False) -> Model:
        reduction_axis = tuple(range(2, len(self.autoencoder.output_shape)))
        predictions_inputs = [self.autoencoder(self.autoencoder.input), self.autoencoder.input]
        predictions = RawPredictionsLayer(reduction_axis)(predictions_inputs)

        inputs = [self.autoencoder.input]
        outputs = [predictions]

        if include_labels_io:
            labels_input_layer = Input(shape=[None, 2], dtype=tf.float32, name="labels_input_layer")
            labels_output_layer = Lambda(tf.identity, name="labels_identity")(labels_input_layer)

            inputs.append(labels_input_layer)
            outputs.append(labels_output_layer)

        predictions_model = Model(inputs=inputs, outputs=outputs, name="predictions_model")
        return predictions_model

    @property
    def raw_predictions_model(self):
        if self._raw_predictions_model is None:
            self._raw_predictions_model = self.make_raw_predictions_model(include_labels_io=True)
        return self._raw_predictions_model

    def predict_anomalies_on_sample(self,
                                    subset: SubsetLoader,
                                    modality: Type[Modality],
                                    sample_index: int,
                                    stride: int,
                                    normalize_predictions=False,
                                    max_steps_count=100000
                                    ):
        dataset = subset.get_source_browser(sample_index, modality, stride)
        predictions, labels = self.raw_predictions_model.predict(dataset, steps=max_steps_count)
        labels = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
        labels = np.any(labels, axis=-1)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / predictions.max()

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    modality: Type[Modality],
                                    stride: int,
                                    max_samples=10):
        predictions, labels = [], []

        sample_count = min(max_samples, len(subset.subset_folders)) if max_samples > 0 else len(subset.subset_folders)
        print("Making predictions for {} videos".format(sample_count))

        for sample_index in range(sample_count):
            sample_name = subset.subset_folders[sample_index]
            print("Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name))
            sample_results = self.predict_anomalies_on_sample(subset, modality,
                                                              sample_index, stride,
                                                              normalize_predictions=False)
            sample_predictions, sample_labels = sample_results
            predictions.append(sample_predictions)
            labels.append(sample_labels)

        return predictions, labels

    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          modality: Type[Modality],
                          stride=1,
                          normalize_predictions=True,
                          max_samples=10):
        predictions, labels = self.predict_anomalies_on_subset(dataset.test_subset,
                                                               modality,
                                                               stride,
                                                               max_samples)

        lengths = np.empty(shape=[len(labels)], dtype=np.int32)
        for i in range(len(labels)):
            lengths[i] = len(labels[i])
            if i > 0:
                lengths[i] += lengths[i - 1]

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        return predictions, labels, lengths

    @staticmethod
    def evaluate_predictions(predictions: np.ndarray,
                             labels: np.ndarray,
                             lengths: np.ndarray = None,
                             output_figure_filepath: str = None,
                             tensorboard: TensorBoard = None,
                             epochs_seen=0):
        if output_figure_filepath is not None:
            plt.plot(np.mean(predictions, axis=1), linewidth=0.5)
            plt.plot(labels, alpha=0.75, linewidth=0.5)
            if lengths is not None:
                lengths_splits = np.zeros(shape=predictions.shape, dtype=np.float32)
                lengths_splits[lengths - 1] = 1.0
                plt.plot(lengths_splits, alpha=0.5, linewidth=0.2)
            plt.savefig(output_figure_filepath, dpi=500)
            plt.clf()  # clf = clear figure

        roc = tf.metrics.AUC(curve="ROC")
        pr = tf.metrics.AUC(curve="PR")

        if predictions.ndim == 2 and labels.ndim == 1:
            predictions = predictions.mean(axis=-1)

        roc.update_state(labels, predictions)
        pr.update_state(labels, predictions)

        roc_result = roc.result()
        pr_result = pr.result()

        if tensorboard is not None:
            # noinspection PyProtectedMember
            with tensorboard._get_writer(tensorboard._train_run_name).as_default():
                tf.summary.scalar(name="Video_ROC_AUC", data=roc_result, step=epochs_seen)
                tf.summary.scalar(name="Video_PR_AUC", data=pr_result, step=epochs_seen)

        return roc_result, pr_result

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             modality: Type[Modality],
                             log_dir: str,
                             stride=1,
                             tensorboard: TensorBoard = None,
                             epochs_seen=0,
                             max_samples=-1,
                             ):
        predictions, labels, lengths = self.predict_anomalies(dataset=dataset,
                                                              modality=modality,
                                                              stride=stride,
                                                              normalize_predictions=True,
                                                              max_samples=max_samples)
        graph_filepath = os.path.join(log_dir, "Anomaly_score.png")
        roc, pr = self.evaluate_predictions(predictions=predictions,
                                            labels=labels,
                                            lengths=lengths,
                                            output_figure_filepath=graph_filepath,
                                            tensorboard=tensorboard,
                                            epochs_seen=epochs_seen)
        print("Anomaly_score : ROC = {} | PR = {}".format(roc, pr))
        return roc, pr


# endregion

def get_autoencoder_loss(input_sequence_length):
    reconstruction_loss_weights = np.ones([input_sequence_length], dtype=np.float32)
    start = 0.5
    stop = 0.1
    step = (stop - start) / input_sequence_length
    prediction_loss_weights = np.arange(start=start, stop=stop, step=step, dtype=np.float32)
    loss_weights = np.concatenate([reconstruction_loss_weights, prediction_loss_weights])
    loss_weights *= (input_sequence_length * 2) / np.sum(loss_weights)
    temporal_loss_weights = tf.constant(loss_weights, name="temporal_loss_weights")

    def autoencoder_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=2)
        reconstruction_loss *= temporal_loss_weights
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    return autoencoder_loss


# region Video

# region Layers
class VideoEncoderLayer(Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        self.conv = Conv3D(filters=filters,
                           padding="same",
                           # basic_block_count=1,
                           kernel_size=kernel_size,
                           kernel_initializer="he_normal",
                           activation=LeakyReLU(alpha=0.3),
                           )
        if strides == 1:
            self.down_sampling = None
        else:
            self.down_sampling = AveragePooling3D(pool_size=strides)

        super(VideoEncoderLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        layer = self.conv(inputs)
        if self.down_sampling is not None:
            layer = self.down_sampling(layer)
        return layer

    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv.compute_output_shape(input_shape)

        if self.down_sampling is not None:
            output_shape = self.down_sampling.compute_output_shape(conv_output_shape)
        else:
            output_shape = conv_output_shape

        return output_shape


class VideoDecoderLayer(Layer):
    def __init__(self, filters, kernel_size, strides, activation=None, **kwargs):
        super(VideoDecoderLayer, self).__init__(**kwargs)

        if activation is None:
            activation = LeakyReLU(alpha=0.3)
        self.deconv = Conv3DTranspose(filters=filters,
                                      padding="same",
                                      # basic_block_count=1,
                                      kernel_size=kernel_size,
                                      kernel_initializer="he_normal",
                                      strides=strides,
                                      activation=activation,
                                      )

    def build(self, input_shape):
        self.deconv.build(input_shape)

    def call(self, inputs, **kwargs):
        layer = self.deconv(inputs)
        return layer

    def compute_output_shape(self, input_shape):
        return self.deconv.compute_output_shape(input_shape)


# endregion

# region Encoder/Decoder
def make_video_encoder(input_shape: Tuple[int, ...]):
    input_layer = Input(input_shape)
    layer = input_layer

    layer = VideoEncoderLayer(filters=32, kernel_size=3, strides=2)(layer)
    layer = VideoEncoderLayer(filters=32, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoEncoderLayer(filters=32, kernel_size=3, strides=1)(layer)

    layer = VideoEncoderLayer(filters=64, kernel_size=3, strides=2)(layer)
    layer = VideoEncoderLayer(filters=64, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=64, kernel_size=3, strides=(1, 2, 2))(layer)

    layer = VideoEncoderLayer(filters=128, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=128, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=128, kernel_size=3, strides=1)(layer)

    output_layer = layer

    encoder = Model(inputs=input_layer, outputs=output_layer, name="encoder")
    return encoder


def make_video_decoder(input_shape: Tuple[int, ...], name, channels_count):
    input_layer = Input(input_shape)
    layer = input_layer

    layer = VideoDecoderLayer(filters=128, kernel_size=3, strides=1)(layer)
    layer = VideoDecoderLayer(filters=128, kernel_size=3, strides=1)(layer)
    layer = VideoDecoderLayer(filters=128, kernel_size=3, strides=1)(layer)

    layer = VideoDecoderLayer(filters=64, kernel_size=3, strides=2)(layer)
    layer = VideoDecoderLayer(filters=64, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoDecoderLayer(filters=64, kernel_size=3, strides=1)(layer)

    layer = VideoDecoderLayer(filters=32, kernel_size=3, strides=2)(layer)
    layer = VideoDecoderLayer(filters=32, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoDecoderLayer(filters=32, kernel_size=3, strides=1)(layer)
    layer = Conv3D(filters=channels_count, kernel_size=1, padding="same")(layer)

    output_layer = layer

    decoder = Model(inputs=input_layer, outputs=output_layer, name=name)
    return decoder


# endregion

def make_video_autoencoder(channels_count=3):
    input_shape = (16, 128, 128, channels_count)

    encoder = make_video_encoder(input_shape)
    encoder_output_shape = encoder.compute_output_shape((None, *input_shape))
    encoder_output_shape = encoder_output_shape[1:]

    reconstructor = make_video_decoder(encoder_output_shape, "video_reconstructor", channels_count)
    reconstructed = reconstructor(encoder(encoder.input))

    predictor = make_video_decoder(encoder_output_shape, "video_predictor", channels_count)
    predicted = predictor(encoder(encoder.input))

    decoded = Concatenate(axis=1)([reconstructed, predicted])
    decoded = Activation("sigmoid")(decoded)

    autoencoder = Model(inputs=encoder.input, outputs=decoded, name="video_autoencoder")
    autoencoder.compile(Adam(lr=1e-4, decay=1e-5), loss=get_autoencoder_loss(input_shape[0]))

    return autoencoder


def train_video_autoencoder():
    channels_count = 3
    video_autoencoder = make_video_autoencoder(channels_count=channels_count)

    # region dataset
    dataset_config = DatasetConfig("C:/datasets/emoly",
                                   modalities_io_shapes=
                                   {
                                       RawVideo: ModalityShape((16, 128, 128, channels_count),
                                                               (32, 128, 128, channels_count))
                                   },
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    # region train
    train_subset = dataset_loader.train_subset
    dataset = train_subset.tf_dataset

    batch_size = 4
    dataset = dataset.batch(batch_size).prefetch(1)
    # endregion

    test_subset = dataset_loader.test_subset
    # dataset_loader.test_subset.subset_folders = [folder for folder in dataset_loader.test_subset.subset_folders
    #                                              if "induced" in folder]
    # endregion

    # region callbacks
    log_dir = "../logs/tests/kitchen_sink"
    log_dir = os.path.join(log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)
    train_image_callbacks = ImageCallback.make_video_autoencoder_callbacks(autoencoder=video_autoencoder,
                                                                           subset=train_subset,
                                                                           name="train",
                                                                           is_train_callback=True,
                                                                           tensorboard=tensorboard,
                                                                           epoch_freq=1)
    test_image_callbacks = ImageCallback.make_video_autoencoder_callbacks(autoencoder=video_autoencoder,
                                                                          subset=test_subset,
                                                                          name="test",
                                                                          is_train_callback=False,
                                                                          tensorboard=tensorboard,
                                                                          epoch_freq=1)
    image_callbacks = train_image_callbacks + test_image_callbacks
    # model_checkpoint = ModelCheckpoint("../logs/tests/kitchen_sink/weights.{epoch:03d}.hdf5",)
    callbacks = [tensorboard, *image_callbacks, TerminateOnNaN()]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(video_autoencoder.name))
    with open(summary_filename, "w") as file:
        video_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))
    # endregion

    # video_autoencoder.load_weights("../logs/tests/kitchen_sink/weights.012.hdf5")

    video_autoencoder.fit(dataset, epochs=500, steps_per_epoch=1000, callbacks=callbacks)

    anomaly_detector = AnomalyDetector(video_autoencoder)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          modality=RawVideo,
                                          log_dir=log_dir,
                                          stride=2)


# endregion


# region Audio

# region Layers
class AudioEncoderLayer(Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", **kwargs):
        self.conv = Conv2D(filters=filters,
                           padding=padding,
                           # basic_block_count=1,
                           kernel_size=kernel_size,
                           kernel_initializer="he_normal",
                           activation=LeakyReLU(alpha=0.3),
                           )
        if strides == 1:
            self.down_sampling = None
        else:
            self.down_sampling = AveragePooling2D(pool_size=strides)

        super(AudioEncoderLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        layer = self.conv(inputs)
        if self.down_sampling is not None:
            layer = self.down_sampling(layer)
        return layer

    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv.compute_output_shape(input_shape)

        if self.down_sampling is not None:
            output_shape = self.down_sampling.compute_output_shape(conv_output_shape)
        else:
            output_shape = conv_output_shape

        return output_shape


class AudioDecoderLayer(Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", activation=None, **kwargs):
        super(AudioDecoderLayer, self).__init__(**kwargs)

        if activation is None:
            activation = LeakyReLU(alpha=0.3)
        self.deconv = Conv2DTranspose(filters=filters,
                                      padding=padding,
                                      # basic_block_count=1,
                                      kernel_size=kernel_size,
                                      kernel_initializer="he_normal",
                                      strides=strides,
                                      activation=activation,
                                      )

    def build(self, input_shape):
        self.deconv.build(input_shape)

    def call(self, inputs, **kwargs):
        layer = self.deconv(inputs)
        return layer

    def compute_output_shape(self, input_shape):
        return self.deconv.compute_output_shape(input_shape)


# endregion

# region Encoder/Decoder
def make_audio_encoder(input_shape: Tuple[int, ...]):
    sequence_length = input_shape[0]
    input_layer = Input(input_shape)
    layer = input_layer

    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=256, kernel_initializer="he_normal", return_sequences=False)(layer)
    layer = RepeatVector(n=sequence_length)(layer)

    output_layer = layer

    encoder = Model(inputs=input_layer, outputs=output_layer, name="encoder")
    return encoder


def make_audio_decoder(input_shape: Tuple[int, ...], name, sequence_length, n_mel_filters):
    input_layer = Input(input_shape)
    layer = input_layer

    layer = CuDNNLSTM(units=256, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=128, kernel_initializer="he_normal", return_sequences=True)(layer)
    layer = Activation("elu")(layer)
    layer = CuDNNLSTM(units=n_mel_filters, kernel_initializer="he_normal", return_sequences=True)(layer)

    output_layer = layer

    decoder = Model(inputs=input_layer, outputs=output_layer, name=name)
    return decoder


# endregion


def run_audio_wave_test(audio_autoencoder: Model,
                        dataset: tf.data.Dataset,
                        max_sample=8,
                        sample_rate=48000,
                        window_length=2400,
                        window_step=1200,
                        n_mel_filters=100,
                        iterations=200):
    from modalities.MelSpectrogram import mel_spectrogram_to_wave
    import librosa

    batch = None
    for batch in dataset:
        break

    batch = batch[0]
    if max_sample == -1:
        batch = batch[:max_sample]

    decoded = audio_autoencoder.predict(batch)

    for i in range(len(batch)):
        wave = mel_spectrogram_to_wave(batch[i], sample_rate, window_length, window_step, n_mel_filters,
                                       iterations=iterations)
        librosa.output.write_wav("../logs/original_{}.wav".format(i), wave, 48000)
        wave = mel_spectrogram_to_wave(decoded[i], sample_rate, window_length, window_step, n_mel_filters,
                                       iterations=iterations)
        librosa.output.write_wav("../logs/decoded_{}.wav".format(i), wave, 48000)


def make_audio_autoencoder(sequence_length, n_mel_filters, add_predictor=False):
    input_shape = (sequence_length, n_mel_filters)

    encoder = make_audio_encoder(input_shape)
    encoder_output_shape = encoder.compute_output_shape((None, *input_shape))
    encoder_output_shape = encoder_output_shape[1:]

    reconstructor = make_audio_decoder(encoder_output_shape, "audio_reconstructor", sequence_length, n_mel_filters)
    reconstructed = reconstructor(encoder(encoder.input))

    if add_predictor:
        predictor = make_audio_decoder(encoder_output_shape, "audio_predictor", sequence_length, n_mel_filters)
        predicted = predictor(encoder(encoder.input))

        decoded = Concatenate(axis=1)([reconstructed, predicted])
    else:
        decoded = reconstructed
    # decoded = Activation("tanh")(decoded)

    autoencoder = Model(inputs=encoder.input, outputs=decoded, name="audio_autoencoder")
    loss = get_autoencoder_loss(sequence_length) if add_predictor else "mse"
    autoencoder.compile(Adam(lr=2e-4, decay=0.0), loss=loss)

    return autoencoder


def train_audio_autoencoder():
    from modalities import MelSpectrogram

    n_mel_filters = 100
    sequence_length = 64
    batch_size = 8

    audio_autoencoder = make_audio_autoencoder(sequence_length=sequence_length,
                                               n_mel_filters=n_mel_filters,
                                               add_predictor=False)
    # audio_autoencoder.load_weights("../logs/tests/kitchen_sink/mfcc_only/weights_020.hdf5")

    dataset_config = DatasetConfig("C:/datasets/emoly",
                                   modalities_io_shapes=
                                   {
                                       MelSpectrogram: ModalityShape((sequence_length, n_mel_filters),
                                                                     (sequence_length, n_mel_filters))
                                   },
                                   output_range=(-1.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    train_subset = dataset_loader.train_subset
    train_subset.subset_folders = [folder for folder in train_subset.subset_folders if "normal" in folder]

    # run_audio_wave_test(audio_autoencoder, dataset, n_mel_filters=n_mel_filters)

    test_subset = dataset_loader.test_subset
    # test_subset.subset_folders = [folder for folder in test_subset.subset_folders if "induced" in folder]

    # region callbacks
    log_dir = "../logs/tests/kitchen_sink/mfcc_only"
    log_dir = os.path.join(log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    train_image_callbacks = ImageCallback.make_image_autoencoder_callbacks(autoencoder=audio_autoencoder,
                                                                           subset=train_subset,
                                                                           name="train",
                                                                           is_train_callback=True,
                                                                           tensorboard=tensorboard,
                                                                           epoch_freq=1)

    raw_predictions = RawPredictionsLayer((2,))([audio_autoencoder.output, audio_autoencoder.input])
    raw_predictions_model = Model(inputs=audio_autoencoder.input, outputs=raw_predictions,
                                  name="raw_predictions_model")
    auc_callback = make_auc_callback(test_subset=test_subset, predictions_model=raw_predictions_model,
                                     tensorboard=tensorboard)

    # model_checkpoint = ModelCheckpoint("../logs/tests/kitchen_sink/mfcc_only/weights.{epoch:03d}.hdf5", )
    model_checkpoint = TmpModelCheckpoint()
    callbacks = [tensorboard, *train_image_callbacks, auc_callback, TerminateOnNaN(), model_checkpoint]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(audio_autoencoder.name))
    with open(summary_filename, "w") as file:
        audio_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))
    # endregion

    train_dataset = train_subset.tf_dataset
    train_dataset = train_dataset.batch(batch_size).prefetch(-1)

    test_dataset = test_subset.tf_dataset
    test_dataset = test_dataset.batch(batch_size)
    audio_autoencoder.fit(train_dataset, epochs=500, steps_per_epoch=10000,
                          validation_data=test_dataset, validation_steps=1000,
                          callbacks=callbacks)

    anomaly_detector = AnomalyDetector(audio_autoencoder)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          modality=MelSpectrogram,
                                          log_dir=log_dir,
                                          stride=1)


# endregion


def main():
    train_audio_autoencoder()


if __name__ == "__main__":
    main()
