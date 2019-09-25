import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.callbacks import TensorBoard, TerminateOnNaN
import os
from time import time
from typing import Tuple

from callbacks import ImageCallback
from datasets.loaders import DatasetConfig, DatasetLoader
from modalities import ModalityLoadInfo, RawVideo, Pattern
from anomaly_detection import AnomalyDetector
from z_kitchen_sink.utils import get_autoencoder_loss


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

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        conv_output_shape = self.conv.compute_output_shape(input_shape)

        if self.down_sampling is not None:
            output_shape = self.down_sampling.compute_output_shape(conv_output_shape)
        else:
            output_shape = conv_output_shape

        return output_shape

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(self.compute_output_shape(input_signature.shape), input_signature.dtype)


# noinspection DuplicatedCode
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

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(self.compute_output_shape(input_signature.shape), input_signature.dtype)


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
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
    autoencoder.compile(optimizer, loss=get_autoencoder_loss(input_shape[0]))

    return autoencoder


def train_video_autoencoder():
    input_length = 16
    output_length = input_length * 2
    channels_count = 3
    video_autoencoder = make_video_autoencoder(channels_count=channels_count)

    # region Dataset
    pattern = Pattern(
        ModalityLoadInfo(RawVideo, input_length),
        ModalityLoadInfo(RawVideo, output_length)
    )
    dataset_config = DatasetConfig("../datasets/emoly", output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    # region Train subset
    train_subset = dataset_loader.train_subset
    dataset = train_subset.make_tf_dataset(pattern)

    batch_size = 4
    dataset = dataset.batch(batch_size).prefetch(1)
    # endregion

    test_subset = dataset_loader.test_subset
    # dataset_loader.test_subset.subset_folders = [folder for folder in dataset_loader.test_subset.subset_folders
    #                                              if "induced" in folder]
    # endregion

    # region Log dir
    log_dir = "../logs/tests/z_kitchen_sink"
    log_dir = os.path.join(log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    # endregion

    # region Callbacks
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)
    train_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_autoencoder,
                                                                subset=train_subset,
                                                                pattern=pattern,
                                                                name="train",
                                                                is_train_callback=True,
                                                                tensorboard=tensorboard,
                                                                epoch_freq=1)
    test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_autoencoder,
                                                               subset=test_subset,
                                                               pattern=pattern,
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

    anomaly_detector = AnomalyDetector(autoencoder=video_autoencoder,
                                       output_length=output_length)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=pattern,
                                          log_dir=log_dir,
                                          stride=2)
