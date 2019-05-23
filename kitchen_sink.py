import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation, Lambda
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from typing import Tuple

from callbacks import ImageCallback, AUCCallback
from datasets.loaders import DatasetConfig, DatasetLoader, SubsetLoader
from modalities import ModalityShape, RawVideo
from layers.utility_layers import RawPredictionsLayer


# region Callbacks
def make_auc_callback(test_subset: SubsetLoader,
                      predictions_model: Model,
                      tensorboard: TensorBoard
                      ) -> AUCCallback:
    inputs, outputs, labels = test_subset.get_batch(batch_size=1024, output_labels=True)
    inputs = inputs[0]
    labels = SubsetLoader.timestamps_labels_to_frame_labels(labels, inputs.shape[1])

    auc_callback = AUCCallback(predictions_model=predictions_model,
                               tensorboard=tensorboard,
                               images=inputs,
                               labels=labels,
                               epoch_freq=1,
                               plot_size=(128, 128),
                               batch_size=16)
    return auc_callback


# endregion

# region Super Test
def make_raw_predictions_model(autoencoder: Model,
                               include_labels_io=False) -> Model:
    reduction_axis = tuple(range(2, len(autoencoder.output_shape)))
    predictions = RawPredictionsLayer(reduction_axis)([autoencoder(autoencoder.input), autoencoder.input])

    inputs = [autoencoder.input]
    outputs = [predictions]

    if include_labels_io:
        labels_input_layer = Input(shape=[None, 2], dtype=tf.float32, name="labels_input_layer")
        labels_output_layer = Lambda(tf.identity, name="labels_identity")(labels_input_layer)

        inputs.append(labels_input_layer)
        outputs.append(labels_output_layer)

    predictions_model = Model(inputs=inputs, outputs=outputs, name="predictions_model")
    return predictions_model


def predict_anomalies_on_video(autoencoder: Model,
                               subset: SubsetLoader,
                               video_index: int,
                               stride: int,
                               normalize_predictions=False,
                               max_steps_count=100000):
    raw_predictions_model = make_raw_predictions_model(autoencoder, include_labels_io=True)
    iterator = subset.get_source_browser(video_index, RawVideo, stride)
    predictions, labels = raw_predictions_model.predict(iterator, steps=max_steps_count)
    labels = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
    labels = np.any(labels, axis=-1)

    if normalize_predictions:
        predictions = (predictions - predictions.min()) / predictions.max()

    return predictions, labels


def predict_anomalies_on_subset(autoencoder: Model,
                                subset: SubsetLoader,
                                stride: int,
                                max_videos=10):
    predictions, labels = [], []

    video_count = min(max_videos, len(subset.subset_folders)) if max_videos > 0 else len(subset.subset_folders)
    print("Making predictions for {} videos".format(video_count))

    for video_index in range(video_count):
        video_name = subset.subset_folders[video_index]
        print("Predicting on video n{}/{} ({})".format(video_index + 1, video_count, video_name))
        video_results = predict_anomalies_on_video(autoencoder, subset, video_index, stride,
                                                   normalize_predictions=False)
        video_predictions, video_labels = video_results
        predictions.append(video_predictions)
        labels.append(video_labels)

    return predictions, labels


def predict_anomalies(autoencoder: Model,
                      dataset: DatasetLoader,
                      stride=1,
                      normalize_predictions=True,
                      max_videos=10):
    predictions, labels = predict_anomalies_on_subset(autoencoder,
                                                      dataset.test_subset,
                                                      stride,
                                                      max_videos)

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


def predict_and_evaluate(autoencoder: Model,
                         dataset: DatasetLoader,
                         log_dir: str,
                         stride=1,
                         tensorboard: TensorBoard = None,
                         epochs_seen=0
                         ):
    predictions, labels, lengths = predict_anomalies(autoencoder=autoencoder,
                                                     dataset=dataset,
                                                     stride=stride,
                                                     normalize_predictions=True,
                                                     max_videos=-1)
    graph_filepath = os.path.join(log_dir, "Anomaly_score.png")
    roc, pr = evaluate_predictions(predictions=predictions,
                                   labels=labels,
                                   lengths=lengths,
                                   output_figure_filepath=graph_filepath,
                                   tensorboard=tensorboard,
                                   epochs_seen=epochs_seen)
    print("Anomaly_score : ROC = {} | PR = {}".format(roc, pr))
    return roc, pr


# endregion

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


def make_video_encoder_small(input_shape: Tuple[int, ...]):
    input_layer = Input(input_shape)
    layer = input_layer

    layer = VideoEncoderLayer(filters=4, kernel_size=3, strides=2)(layer)
    layer = VideoEncoderLayer(filters=4, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoEncoderLayer(filters=4, kernel_size=3, strides=1)(layer)

    layer = VideoEncoderLayer(filters=8, kernel_size=3, strides=2)(layer)
    layer = VideoEncoderLayer(filters=8, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=8, kernel_size=3, strides=(1, 2, 2))(layer)

    layer = VideoEncoderLayer(filters=16, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=16, kernel_size=3, strides=1)(layer)
    layer = VideoEncoderLayer(filters=16, kernel_size=3, strides=1)(layer)

    output_layer = layer

    encoder = Model(inputs=input_layer, outputs=output_layer, name="encoder")
    return encoder


def make_video_decoder_small(input_shape: Tuple[int, ...], name, channels_count):
    input_layer = Input(input_shape)
    layer = input_layer

    layer = VideoDecoderLayer(filters=16, kernel_size=3, strides=1)(layer)
    layer = VideoDecoderLayer(filters=16, kernel_size=3, strides=1)(layer)
    layer = VideoDecoderLayer(filters=16, kernel_size=3, strides=1)(layer)

    layer = VideoDecoderLayer(filters=8, kernel_size=3, strides=2)(layer)
    layer = VideoDecoderLayer(filters=8, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoDecoderLayer(filters=8, kernel_size=3, strides=1)(layer)

    layer = VideoDecoderLayer(filters=4, kernel_size=3, strides=2)(layer)
    layer = VideoDecoderLayer(filters=4, kernel_size=3, strides=(1, 2, 2))(layer)
    layer = VideoDecoderLayer(filters=4, kernel_size=3, strides=1)(layer)
    layer = Conv3D(filters=channels_count, kernel_size=1, padding="same")(layer)

    output_layer = layer

    decoder = Model(inputs=input_layer, outputs=output_layer, name=name)
    return decoder


# endregion

# region Autoencoder
def get_autoencoder_loss(input_sequence_length):
    reconstruction_loss_weights = np.ones([input_sequence_length], dtype=np.float32)
    stop = 0.2
    step = (stop - 1) / input_sequence_length
    prediction_loss_weights = np.arange(start=1.0, stop=stop, step=step, dtype=np.float32)
    loss_weights = np.concatenate([reconstruction_loss_weights, prediction_loss_weights])
    loss_weights *= (input_sequence_length * 2) / np.sum(loss_weights)
    temporal_loss_weights = tf.constant(loss_weights, name="temporal_loss_weights")

    def autoencoder_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=(2, 3, 4))
        reconstruction_loss *= temporal_loss_weights
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    return autoencoder_loss


def make_video_autoencoder(channels_count=3):
    input_shape = (16, 128, 128, channels_count)

    encoder = make_video_encoder_small(input_shape)
    encoder_output_shape = encoder.compute_output_shape((None, *input_shape))
    encoder_output_shape = encoder_output_shape[1:]

    reconstructor = make_video_decoder(encoder_output_shape, "reconstructor", channels_count)
    reconstructed = reconstructor(encoder(encoder.input))

    predictor = make_video_decoder_small(encoder_output_shape, "predictor", channels_count)
    predicted = predictor(encoder(encoder.input))

    decoded = Concatenate(axis=1)([reconstructed, predicted])
    decoded = Activation("sigmoid")(decoded)

    autoencoder = Model(inputs=encoder.input, outputs=decoded, name="autoencoder")
    autoencoder.compile(Adam(lr=8e-6, decay=5e-5), loss=get_autoencoder_loss(input_shape[0]))

    return autoencoder


# endregion

def main():
    from modalities import MelSpectrogram
    channels_count = 1
    video_autoencoder = make_video_autoencoder(channels_count=channels_count)

    # region dataset
    dataset_config = DatasetConfig("../datasets/emoly",
                                   modalities_io_shapes=
                                   {
                                       MelSpectrogram: ModalityShape((52, 100),
                                                                     (104, 100))
                                   },
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    # region train
    train_subset = dataset_loader.train_subset
    dataset = train_subset.tf_dataset

    batch_size = 4
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
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

    video_autoencoder.fit(dataset, epochs=500, steps_per_epoch=2, callbacks=callbacks)

    predict_and_evaluate(autoencoder=video_autoencoder,
                         dataset=dataset_loader,
                         log_dir=log_dir,
                         stride=2)


if __name__ == "__main__":
    main()
