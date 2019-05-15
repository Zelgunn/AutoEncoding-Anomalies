import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation, Lambda
from tensorflow.python.keras.layers import Conv1D, AveragePooling1D
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN
from tensorflow.python.keras import backend
import os
from time import time
from typing import Tuple

from callbacks import ImageCallback, RunModel
from utils.summary_utils import image_summary
from datasets.loaders import DatasetConfig, DatasetLoader, SubsetLoader
from modalities import ModalityShape, RawVideo
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1, sampling
from layers import ResBlock3D, ResBlock3DTranspose


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


def make_video_decoder(input_shape: Tuple[int, ...], name):
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
    layer = Conv3D(filters=3, kernel_size=1, padding="same")(layer)

    output_layer = layer

    decoder = Model(inputs=input_layer, outputs=output_layer, name=name)
    return decoder


def make_video_autoencoder():
    input_shape = (16, 128, 128, 3)

    encoder = make_video_encoder(input_shape)
    encoder_output_shape = encoder.compute_output_shape((None, *input_shape))
    encoder_output_shape = encoder_output_shape[1:]

    reconstructor = make_video_decoder(encoder_output_shape, "reconstructor")
    reconstructed = reconstructor(encoder(encoder.input))

    predictor = make_video_decoder(encoder_output_shape, "predictor")
    predicted = predictor(encoder(encoder.input))

    decoded = Concatenate(axis=1)([reconstructed, predicted])
    decoded = Activation("sigmoid")(decoded)

    autoencoder = Model(inputs=encoder.input, outputs=decoded, name="autoencoder")
    autoencoder.compile(Adam(lr=5e-4),
                        loss="mean_squared_error")

    return autoencoder


def make_image_callback(autoencoder: Model,
                        subset: SubsetLoader,
                        name: str,
                        tensorboard: TensorBoard,
                        frequency="epoch",
                        ) -> ImageCallback:
    inputs, outputs = subset.get_batch(batch_size=4, output_labels=False)
    videos = [inputs[0], outputs[0]]

    true_outputs_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[None, 32, 128, 128, 3],
                                              name="true_outputs_placeholder")

    summary_inputs = [autoencoder.input, true_outputs_placeholder]

    def normalize_image_tensor(image_tensor):
        normalized = tf.cast(image_tensor * 255, tf.uint8)
        return normalized

    true_outputs = normalize_image_tensor(true_outputs_placeholder)
    pred_outputs = normalize_image_tensor(autoencoder.output)

    io_delta = (pred_outputs - true_outputs) * (tf.cast(pred_outputs < true_outputs, dtype=tf.uint8) * 254 + 1)

    max_outputs = 4
    one_shot_summaries = [image_summary(name + "_true_outputs", true_outputs, max_outputs, fps=8),
                          image_summary(name + "_true_outputs_full_speed", true_outputs, max_outputs, fps=25)]
    repeated_summaries = [image_summary(name + "_pred_outputs", pred_outputs, max_outputs, fps=8),
                          image_summary(name + "_pred_outputs_full_speed", pred_outputs, max_outputs, fps=25),
                          image_summary(name + "_delta", io_delta, max_outputs, fps=8)]

    one_shot_summary_model = RunModel(summary_inputs, one_shot_summaries, output_is_summary=True)
    repeated_summary_model = RunModel(summary_inputs, repeated_summaries, output_is_summary=True)

    return ImageCallback(one_shot_summary_model, repeated_summary_model, videos, tensorboard, frequency, epoch_freq=1)


def main():
    video_autoencoder = make_video_autoencoder()

    # region dataset
    dataset_config = DatasetConfig("../datasets/emoly",
                                   modalities_io_shapes=
                                   {
                                       RawVideo: ModalityShape((16, 128, 128, 3),
                                                               (32, 128, 128, 3))
                                   },
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    train_loader = dataset_loader.train_subset
    dataset = train_loader.tf_dataset

    batch_size = 4
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_initializable_iterator()
    dataset_initializer = iterator.initializer
    iterator = iterator.get_next()
    x, y = iterator
    x = x[0]
    y = y[0]

    session = backend.get_session()
    session.run(dataset_initializer)
    # endregion

    # region callbacks
    log_dir = "../logs/tests/kitchen_sink"
    log_dir = os.path.join(log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="batch")

    image_callback = make_image_callback(video_autoencoder, train_loader, "image_callback", tensorboard)

    # model_checkpoint = ModelCheckpoint("../logs/tests/kitchen_sink/weights.{epoch:03d}.hdf5")

    callbacks = [tensorboard, image_callback, TerminateOnNaN()]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(video_autoencoder.name))
    with open(summary_filename, "w") as file:
        video_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))
    # endregion

    # video_autoencoder.load_weights("../logs/tests/kitchen_sink/weights.003.hdf5")

    video_autoencoder.fit(x, y, epochs=100, steps_per_epoch=1000, callbacks=callbacks)


if __name__ == "__main__":
    main()
