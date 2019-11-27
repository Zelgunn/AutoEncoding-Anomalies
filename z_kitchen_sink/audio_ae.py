import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation
from tensorflow.python.keras.layers import CuDNNLSTM, RepeatVector
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose
from tensorflow.python.keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
import os
from time import time
from typing import Tuple

from callbacks import ImageCallback, AUCCallback
from datasets.loaders import DatasetConfig, DatasetLoader
from modalities import ModalityLoadInfo, Pattern
from anomaly_detection import AnomalyDetector, IOCompareModel
from z_kitchen_sink.utils import get_autoencoder_loss


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

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(self.compute_output_shape(input_signature.shape), input_signature.dtype)


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

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(self.compute_output_shape(input_signature.shape), input_signature.dtype)


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


def make_audio_decoder(input_shape: Tuple[int, ...], name, _, n_mel_filters):
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
    optimizer = tf.keras.optimizers.Adam(lr=2e-4, decay=0.0)
    autoencoder.compile(optimizer, loss=loss)

    return autoencoder


def train_audio_autoencoder():
    from modalities import MelSpectrogram

    n_mel_filters = 100
    input_length = 64
    output_length = input_length
    batch_size = 8

    audio_autoencoder = make_audio_autoencoder(sequence_length=input_length,
                                               n_mel_filters=n_mel_filters,
                                               add_predictor=False)
    # region Dataset
    # audio_autoencoder.load_weights("../logs/tests/kitchen_sink/mfcc_only/weights_020.hdf5")
    pattern = Pattern(
        ModalityLoadInfo(MelSpectrogram, input_length),
        ModalityLoadInfo(MelSpectrogram, output_length)
    )
    anomaly_pattern = Pattern(
        *pattern,
        "labels"
    )
    dataset_config = DatasetConfig("../datasets/emoly",

                                   output_range=(-1.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    train_subset = dataset_loader.train_subset
    train_subset.subset_folders = [folder for folder in train_subset.subset_folders if "normal" in folder]

    # run_audio_wave_test(audio_autoencoder, dataset, n_mel_filters=n_mel_filters)

    test_subset = dataset_loader.test_subset
    # test_subset.subset_folders = [folder for folder in test_subset.subset_folders if "induced" in folder]
    # endregion

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/mfcc"
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    # endregion

    # region Callbacks
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    train_image_callbacks = ImageCallback.make_image_autoencoder_callbacks(autoencoder=audio_autoencoder,
                                                                           subset=train_subset,
                                                                           pattern=pattern,
                                                                           name="train",
                                                                           is_train_callback=True,
                                                                           tensorboard=tensorboard,
                                                                           epoch_freq=1)

    raw_predictions_model = IOCompareModel(audio_autoencoder)
    auc_callback = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=test_subset, pattern=anomaly_pattern)

    model_checkpoint = ModelCheckpoint(os.path.join(base_log_dir, "weights.{epoch:03d}.hdf5", ))
    callbacks = [tensorboard, *train_image_callbacks, auc_callback, TerminateOnNaN(), model_checkpoint]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(audio_autoencoder.name))
    with open(summary_filename, "w") as file:
        audio_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))
    # endregion

    train_dataset = train_subset.make_tf_dataset(pattern)
    train_dataset = train_dataset.batch(batch_size).prefetch(-1)

    test_dataset = test_subset.make_tf_dataset(pattern)
    test_dataset = test_dataset.batch(batch_size)
    audio_autoencoder.fit(train_dataset, epochs=500, steps_per_epoch=10000,
                          validation_data=test_dataset, validation_steps=1000,
                          callbacks=callbacks)

    anomaly_detector = AnomalyDetector(autoencoder=audio_autoencoder,
                                       output_length=output_length)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=anomaly_pattern,
                                          log_dir=log_dir,
                                          stride=1)
