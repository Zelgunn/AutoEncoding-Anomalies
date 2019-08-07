import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LeakyReLU, Layer, Concatenate, Activation, Lambda, Flatten, Reshape
from tensorflow.python.keras.layers import Conv3D, AveragePooling3D, Conv3DTranspose
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose
from tensorflow.python.keras.layers import Conv1D, AveragePooling1D, UpSampling1D
from tensorflow.python.keras.layers import CuDNNLSTM, RepeatVector, TimeDistributed
from tensorflow.python.keras.layers import Dense, ZeroPadding1D
from tensorflow.python.keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
import numpy as np
import os
import cv2
from time import time
from typing import Tuple, List

from callbacks import ImageCallback, AUCCallback, LandmarksVideoCallback
from datasets.loaders import DatasetConfig, DatasetLoader
from modalities import ModalityLoadInfo, RawVideo, Landmarks, Pattern
from anomaly_detection import RawPredictionsLayer, AnomalyDetector
from utils.train_utils import save_model_info
from transformer import Transformer
from transformer.layers import PositionalEncodingMode


# region Utility
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


# region Helpers
def get_temporal_loss_weights(input_sequence_length, start=1.0, stop=0.1):
    reconstruction_loss_weights = np.ones([input_sequence_length], dtype=np.float32)
    step = (stop - start) / input_sequence_length
    prediction_loss_weights = np.arange(start=start, stop=stop, step=step, dtype=np.float32)
    loss_weights = np.concatenate([reconstruction_loss_weights, prediction_loss_weights])
    loss_weights *= (input_sequence_length * 2) / np.sum(loss_weights)
    temporal_loss_weights = tf.constant(loss_weights, name="temporal_loss_weights")
    return temporal_loss_weights


def get_autoencoder_loss(input_sequence_length):
    temporal_loss_weights = get_temporal_loss_weights(input_sequence_length)

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


# endregion
# endregion

# region Video
# region Video (autoencoder)

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
        ModalityLoadInfo(RawVideo, input_length, (input_length, 128, 128, channels_count)),
        ModalityLoadInfo(RawVideo, output_length, (output_length, 128, 128, channels_count))
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
    log_dir = "../logs/tests/kitchen_sink"
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

    anomaly_detector = AnomalyDetector(inputs=video_autoencoder.inputs,
                                       output=video_autoencoder.output,
                                       ground_truth=video_autoencoder.inputs)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=pattern,
                                          log_dir=log_dir,
                                          stride=2)


# endregion

# region Scaling Video Transformer
def train_scaling_video_transformer():
    from transformer import VideoTransformer

    # tf.keras.mixed_precision.experimental.set_policy("infer_float32_vars")

    dataset_name = "ucsd"
    batch_size = 2
    length = 16
    height = 128
    width = 128
    channel_count = 1 if dataset_name is "ucsd" else 1
    mode = "train"
    initial_epoch = 0

    video_transformer = VideoTransformer(input_shape=(length, height, width, channel_count),
                                         subscale_stride=(4, 2, 2),
                                         embedding_size=64,
                                         hidden_size=256,
                                         block_sizes=[
                                             (4, 8, 8),
                                             (1, 64, 4),
                                             (1, 4, 64),
                                             (4, 8, 8),
                                             # (1, 4, 32),
                                             # (1, 32, 4),
                                             # (4, 4, 8),
                                             # (4, 8, 4)
                                         ],
                                         attention_head_count=8,
                                         attention_head_size=32,
                                         copy_regularization_factor=1.0,
                                         positional_encoding_range=0.1
                                         )

    # region Patterns
    def reduce_channels(video: tf.Tensor):
        return tf.reduce_mean(video, axis=-1, keepdims=True)

    def preprocess_video(video: tf.Tensor):
        video = reduce_channels(video)

        if mode == "train":
            crop_ratio = tf.random.uniform(shape=(), minval=0.75, maxval=1.0)
            crop_size = [length, crop_ratio * height, crop_ratio * width, channel_count]
            video = tf.image.random_crop(video, crop_size)
            video = tf.image.resize(video, (height, width))
            video = tf.image.random_brightness(video, max_delta=0.2)

        return video

    pattern = Pattern(
        ModalityLoadInfo(RawVideo, length, (length, height, width, channel_count), preprocess_video)
    )
    anomaly_pattern = Pattern(*pattern, "labels")
    # endregion

    # region Datasets
    dataset_path = "../datasets/ucsd/ped2" if dataset_name is "ucsd" else "../datasets/emoly"
    dataset_config = DatasetConfig(dataset_path, output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    dataset = dataset_loader.train_subset.make_tf_dataset(pattern)
    dataset = dataset.batch(batch_size)

    # dataset_loader.test_subset.subset_folders = [folder for folder in dataset_loader.test_subset.subset_folders
    #                                              if "induced" in folder]
    validation_dataset = dataset_loader.test_subset.make_tf_dataset(pattern)
    validation_dataset = validation_dataset.batch(batch_size)
    # endregion

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_transformer/{}".format(dataset_name)
    base_log_dir = os.path.normpath(base_log_dir)
    weights_name = "weights_{epoch:03d}.hdf5"
    dir_name = "log_{}" if mode is "train" else "test_log_{}"
    log_dir = os.path.join(base_log_dir, dir_name.format(int(time())))
    if mode is not "show":
        os.makedirs(log_dir)
    save_model_info(video_transformer.trainer, log_dir)
    weights_path = os.path.join(log_dir, weights_name)
    # endregion

    if initial_epoch > 0:
        video_transformer.trainer.load_weights(os.path.join(base_log_dir, weights_name.format(epoch=initial_epoch)))

    if mode == "train":
        # region Callbacks
        callbacks: List[tf.keras.callbacks.Callback] = []

        tensorboard = TensorBoard(log_dir=log_dir, profile_batch=0)
        callbacks.append(tensorboard)

        model_checkpoint = ModelCheckpoint(weights_path, verbose=1)
        callbacks.append(model_checkpoint)

        # region Image callbacks
        train_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_transformer,
                                                                    subset=dataset_loader.train_subset,
                                                                    pattern=pattern,
                                                                    name="train",
                                                                    is_train_callback=True,
                                                                    tensorboard=tensorboard,
                                                                    epoch_freq=1)
        test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_transformer,
                                                                   subset=dataset_loader.test_subset,
                                                                   pattern=pattern,
                                                                   name="test",
                                                                   is_train_callback=False,
                                                                   tensorboard=tensorboard,
                                                                   epoch_freq=1)
        alt_video_transformer = video_transformer.model_using_decoder_outputs
        alt_test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=alt_video_transformer,
                                                                       subset=dataset_loader.test_subset,
                                                                       pattern=pattern,
                                                                       name="test_udo",
                                                                       is_train_callback=False,
                                                                       tensorboard=tensorboard,
                                                                       epoch_freq=1)

        image_callbacks = train_image_callbacks + test_image_callbacks + alt_test_image_callbacks
        callbacks += image_callbacks

        # endregion
        # region AUC
        def make_transformer_auc_callback(use_decoder_outputs: bool):
            if use_decoder_outputs:
                output_used = video_transformer.model_using_decoder_outputs.output
                prefix_used = "udo"  # udo = using decoder outputs
            else:
                output_used = video_transformer.output
                prefix_used = "ugt"  # ugt = using ground truth

            raw_predictions = RawPredictionsLayer(output_length=length)([output_used, video_transformer.input])
            raw_predictions_model = Model(inputs=video_transformer.inputs,
                                          outputs=raw_predictions,
                                          name="{}_raw_predictions_model".format(prefix_used))

            return AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=dataset_loader.test_subset, pattern=anomaly_pattern,
                                           samples_count=128, epoch_freq=1, batch_size=4, prefix=prefix_used)

        ugt_auc_callback = make_transformer_auc_callback(False)
        callbacks.append(ugt_auc_callback)
        udo_auc_callback = make_transformer_auc_callback(True)
        callbacks.append(udo_auc_callback)
        # endregion
        # endregion

        video_transformer.fit(dataset,
                              batch_size=batch_size,
                              epochs=100,
                              steps_per_epoch=300,
                              initial_epoch=initial_epoch,
                              validation_data=validation_dataset,
                              validation_steps=20,
                              callbacks=callbacks)
    elif mode == "show":
        video_transformer.trainer.summary()

        for i, batch in zip(range(10), validation_dataset):
            predicted_easy = video_transformer(batch[:1], use_decoder_outputs=False)
            predicted_hard = video_transformer(batch[:1], use_decoder_outputs=True)
            for k in range(16):
                predicted_frame_easy = predicted_easy.numpy()[0, k]
                predicted_frame_hard = predicted_hard.numpy()[0, k]
                predicted_frame_easy = cv2.resize(predicted_frame_easy, (256, 256), interpolation=cv2.INTER_NEAREST)
                predicted_frame_hard = cv2.resize(predicted_frame_hard, (256, 256), interpolation=cv2.INTER_NEAREST)
                true_frame = cv2.resize(batch.numpy()[0, k], (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("predicted_easy", predicted_frame_easy)
                cv2.imshow("predicted_hard", predicted_frame_hard)
                cv2.imshow("frame", true_frame)
                cv2.waitKey(0)
    elif mode == "anomaly":
        detector_metric = "mse"
        detector_stride = 4
        pre_normalize_predictions = True

        anomaly_detector = AnomalyDetector(inputs=video_transformer.input,
                                           output=video_transformer.output,
                                           ground_truth=video_transformer.input,
                                           metric=detector_metric)

        anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                              pattern=anomaly_pattern.with_added_depth().with_added_depth(),
                                              log_dir=log_dir,
                                              stride=detector_stride,
                                              pre_normalize_predictions=pre_normalize_predictions,
                                              additional_config=
                                              {
                                                  "initial_epoch": initial_epoch
                                              }
                                              )


# endregion

# region Video CNN-Transformer
class VideoCNNTransformerLoss(tf.losses.Loss):
    def __init__(self, beta=1e-3, **kwargs):
        self.beta = beta
        super(VideoCNNTransformerLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true.set_shape(y_pred.shape)

        loss = tf.square(y_true - y_pred)
        shifted_loss = tf.square(y_true[:, :-1] - y_pred[:, 1:])

        loss = tf.reduce_mean(loss, axis=[1, 2, 3, 4])
        shifted_loss = tf.reduce_mean(shifted_loss, axis=[1, 2, 3, 4])

        shift_regularization = tf.sigmoid(tf.math.log(loss / (shifted_loss + 1e-7))) * self.beta

        loss = tf.reduce_mean(loss)
        shift_regularization = tf.reduce_mean(shift_regularization)

        return loss + shift_regularization


def make_video_cnn_transformer(input_length: int, output_length: int, height: int, width: int, channels: int) -> Model:
    input_shape = (output_length, height, width, channels)
    input_layer = Input(input_shape, name="input_layer")

    common_cnn_params = {"kernel_size": 3,
                         "padding": "same",
                         "activation": "elu",
                         "kernel_initializer": "he_normal"}
    # region Encoder
    cnn_encoder_layers = [
        Conv2D(filters=4, strides=1, **common_cnn_params),
        Conv2D(filters=4, strides=2, **common_cnn_params),

        Conv2D(filters=8, strides=1, **common_cnn_params),
        Conv2D(filters=8, strides=2, **common_cnn_params),

        Conv2D(filters=16, strides=1, **common_cnn_params),
        Conv2D(filters=16, strides=2, **common_cnn_params),

        Conv2D(filters=32, strides=1, **common_cnn_params),
        Conv2D(filters=32, strides=2, **common_cnn_params),

        Conv2D(filters=64, strides=1, **common_cnn_params),
        Conv2D(filters=64, strides=2, **common_cnn_params),
    ]
    x = input_layer
    for layer in cnn_encoder_layers:
        x = TimeDistributed(layer)(x)
    encoder_output_shape = x.shape[-3:]

    x = TimeDistributed(Flatten())(x)
    cnn_encoder_output = x
    units = cnn_encoder_output.shape[-1]
    # endregion

    # region Transformer
    frame_per_prediction = 1
    transformer = Transformer(max_input_length=input_length,
                              max_output_length=output_length // frame_per_prediction,
                              input_size=units,
                              output_size=units * frame_per_prediction,
                              output_activation="linear",
                              layers_intermediate_size=32,
                              layers_count=4,
                              attention_heads_count=4,
                              attention_key_size=16,
                              attention_values_size=16,
                              dropout_rate=0.0,
                              # decoder_pre_net=pre_net,
                              decoder_pre_net=None,
                              positional_encoding_mode=PositionalEncodingMode.ADD,
                              positional_encoding_range=0.1,
                              positional_encoding_size=32,
                              name="Transformer")

    transformer_encoder_inputs = Lambda(lambda inputs: inputs[:, :input_length], name="transformer_encoder_inputs")(x)
    shifted_x = Lambda(lambda inputs: inputs[:, :-1], name="shift_predictions")(x)
    transformer_decoder_inputs = ZeroPadding1D(padding=(1, 0), name="predictions")(shifted_x)

    x = transformer([transformer_encoder_inputs, transformer_decoder_inputs])
    # endregion

    # region Decoder
    x = TimeDistributed(Reshape(encoder_output_shape))(x)

    cnn_decoder_layers = [
        Conv2DTranspose(filters=64, strides=2, **common_cnn_params),

        Conv2DTranspose(filters=32, strides=1, **common_cnn_params),
        Conv2DTranspose(filters=32, strides=2, **common_cnn_params),

        Conv2DTranspose(filters=16, strides=1, **common_cnn_params),
        Conv2DTranspose(filters=16, strides=2, **common_cnn_params),

        Conv2DTranspose(filters=8, strides=1, **common_cnn_params),
        Conv2DTranspose(filters=8, strides=2, **common_cnn_params),

        Conv2DTranspose(filters=4, strides=1, **common_cnn_params),
        Conv2DTranspose(filters=4, strides=2, **common_cnn_params),

        Conv2DTranspose(filters=channels, strides=1, **common_cnn_params),
    ]

    for layer in cnn_decoder_layers:
        x = TimeDistributed(layer)(x)
    # endregion

    model = Model(inputs=input_layer, outputs=x, name="video_cnn_transformer")
    model.compile("adam", VideoCNNTransformerLoss())

    return model


def train_video_cnn_transformer():
    model = make_video_cnn_transformer(16, 32, 128, 128, 1)
    model.summary()

    # region Dataset
    pattern = Pattern(
        ModalityLoadInfo(RawVideo, 32, (32, 128, 128, 1)),
        ModalityLoadInfo(RawVideo, 32, (32, 128, 128, 1)),
    )

    dataset_config = DatasetConfig("../datasets/ucsd/ped2", output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    dataset = dataset_loader.train_subset.make_tf_dataset(pattern)
    dataset = dataset.batch(16).prefetch(-1)
    # endregion

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_cnn_transformer"
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(model, log_dir)
    weights_path = os.path.join(log_dir, "weights_{epoch:03d}.hdf5")
    # endregion

    # region Callbacks
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)
    train_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=model,
                                                                subset=dataset_loader.train_subset,
                                                                pattern=pattern,
                                                                name="train",
                                                                is_train_callback=True,
                                                                tensorboard=tensorboard,
                                                                epoch_freq=1)
    test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=model,
                                                               subset=dataset_loader.test_subset,
                                                               pattern=pattern,
                                                               name="test",
                                                               is_train_callback=False,
                                                               tensorboard=tensorboard,
                                                               epoch_freq=1)
    image_callbacks = train_image_callbacks + test_image_callbacks
    model_checkpoint = TmpModelCheckpoint(weights_path)
    callbacks = [tensorboard, *image_callbacks, model_checkpoint]
    # endregion

    model.fit(dataset, steps_per_epoch=50, epochs=5, callbacks=callbacks)


# endregion
# endregion

# region Landmarks
def get_landmarks_datasets():
    dataset_config = DatasetConfig("../datasets/emoly",
                                   output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)
    train_subset = dataset_loader.train_subset
    train_subset.subset_folders = [folder for folder in train_subset.subset_folders if "normal" in folder]

    test_subset = dataset_loader.test_subset
    test_subset.subset_folders = [folder for folder in test_subset.subset_folders if "induced" in folder]

    return dataset_loader, train_subset, test_subset


# region Landmarks (autoencoder)

def make_landmarks_encoder(input_layer, sequence_length):
    common_args = {"kernel_size": 3,
                   "padding": "same",
                   "activation": "relu",
                   "use_bias": False}

    layer = input_layer
    layer = Reshape([sequence_length, 68 * 2])(layer)

    layer = Conv1D(filters=64, **common_args)(layer)
    layer = Conv1D(filters=64, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Flatten()(layer)
    layer = Dense(units=64, activation="relu")(layer)

    return layer


def make_landmarks_decoder(input_layer, sequence_length):
    common_args = {"kernel_size": 3,
                   "padding": "same",
                   "activation": "relu",
                   "use_bias": False}

    layer = input_layer

    layer = Dense(units=8 * 16, activation="relu")(layer)
    layer = Reshape([8, 16])(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=64, **common_args)(layer)
    layer = Conv1D(filters=68 * 2, kernel_size=3, padding="same", activation=None, use_bias=False)(layer)

    layer = Reshape([sequence_length, 68, 2])(layer)

    return layer


def make_landmarks_autoencoder(sequence_length: int, add_predictor: bool):
    input_shape = (sequence_length, 68, 2)

    input_layer = Input(input_shape)

    encoded = make_landmarks_encoder(input_layer, sequence_length)
    decoded = make_landmarks_decoder(encoded, sequence_length)
    if add_predictor:
        predicted = make_landmarks_decoder(encoded, sequence_length)
        decoded = Concatenate(axis=1)([decoded, predicted])

    output_layer = decoded

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="landmarks_autoencoder")

    # loss = get_autoencoder_loss(sequence_length, axis=(2, 3)) if add_predictor else "mse"

    def landmarks_loss(y_true, y_pred):
        indexes = [36, 39, 42, 45] + list(range(48, 68))

        mask = []
        for i in range(68):
            mask.append(1.0 if i in indexes else 1e-1)

        mask = tf.constant(mask, shape=[1, 1, 68, 1])
        loss = tf.square(y_true - y_pred) * mask
        loss = tf.reduce_mean(loss, axis=[2, 3])
        loss *= get_temporal_loss_weights(sequence_length)
        loss = tf.reduce_mean(loss, axis=1)
        return loss

    optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=0.0)
    autoencoder.compile(optimizer, loss=landmarks_loss)

    return autoencoder


def train_landmarks_autoencoder():
    add_predictor = True
    input_length = 64
    output_length = input_length * 2 if add_predictor else input_length
    batch_size = 16

    landmarks_autoencoder = make_landmarks_autoencoder(sequence_length=input_length,
                                                       add_predictor=add_predictor)
    # landmarks_autoencoder.load_weights("../logs/tests/kitchen_sink/mfcc_only/weights_020.hdf5")

    dataset_loader, train_subset, test_subset = get_landmarks_datasets()
    pattern = Pattern(
        ModalityLoadInfo(Landmarks, input_length, (input_length, 136)),
        ModalityLoadInfo(Landmarks, output_length, (output_length, 136))
    )

    train_dataset = train_subset.make_tf_dataset(pattern)
    train_dataset = train_dataset.batch(batch_size).prefetch(-1)

    test_dataset = test_subset.make_tf_dataset(pattern)
    test_dataset = test_dataset.batch(batch_size)

    # region Callbacks
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/landmarks"
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    raw_predictions = RawPredictionsLayer()([landmarks_autoencoder.output, landmarks_autoencoder.input])
    raw_predictions_model = Model(inputs=landmarks_autoencoder.input, outputs=raw_predictions,
                                  name="raw_predictions_model")
    auc_callback = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=test_subset, pattern=Pattern(*pattern, "labels"))

    train_landmarks_callback = LandmarksVideoCallback(train_subset, landmarks_autoencoder, tensorboard, pattern=pattern)
    test_landmarks_callback = LandmarksVideoCallback(test_subset, landmarks_autoencoder, tensorboard, pattern=pattern)

    model_checkpoint = TmpModelCheckpoint(os.path.join(base_log_dir, "weights_{epoch:03d}.hdf5"))
    callbacks = [tensorboard,
                 train_landmarks_callback,
                 test_landmarks_callback,
                 auc_callback,
                 model_checkpoint]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(landmarks_autoencoder.name))
    with open(summary_filename, "w") as file:
        landmarks_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))

    # endregion

    landmarks_autoencoder.fit(train_dataset, epochs=10, steps_per_epoch=10000,
                              validation_data=test_dataset, validation_steps=200,
                              callbacks=callbacks)
    # landmarks_autoencoder.load_weights(os.path.join(base_log_dir, "weights_006.hdf5"))

    anomaly_detector = AnomalyDetector(inputs=landmarks_autoencoder.inputs,
                                       output=landmarks_autoencoder.output,
                                       ground_truth=landmarks_autoencoder.inputs)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=pattern,
                                          log_dir=log_dir,
                                          stride=1)


# endregion

# region Landmarks (transformer)

def add_batch_noise(*args):
    inputs, outputs = args[0]
    noise = tf.random.uniform([], minval=-0.1, maxval=0.1)
    inputs = tf.clip_by_value(inputs + noise, clip_value_min=0.0, clip_value_max=1.0)
    outputs = tf.clip_by_value(outputs + noise, clip_value_min=0.0, clip_value_max=1.0)
    return (inputs, outputs),


def train_landmarks_transformer():
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/landmarks_transformer"
    batch_size = 16
    input_length = 32
    output_length = input_length + 16
    frame_per_prediction = 4

    landmarks_transformer = Transformer(max_input_length=input_length,
                                        max_output_length=output_length // frame_per_prediction,
                                        input_size=68 * 2,
                                        output_size=68 * 2 * frame_per_prediction,
                                        output_activation="linear",
                                        layers_intermediate_size=32,
                                        layers_count=4,
                                        attention_heads_count=4,
                                        attention_key_size=16,
                                        attention_values_size=16,
                                        dropout_rate=0.0,
                                        # decoder_pre_net=pre_net,
                                        decoder_pre_net=None,
                                        positional_encoding_mode=PositionalEncodingMode.ADD,
                                        positional_encoding_range=0.1,
                                        positional_encoding_size=32,
                                        name="Transformer")

    landmarks_transformer.add_transformer_loss()

    optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.98)
    landmarks_transformer.compile(optimizer)
    landmarks_transformer.summary()

    landmarks_transformer.load_weights(base_log_dir + "/weights_011.hdf5")

    autonomous_transformer = landmarks_transformer.make_autonomous_model()
    optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.98)
    autonomous_transformer.compile(optimizer, loss="mse")

    # region Datasets
    dataset_loader, train_subset, test_subset = get_landmarks_datasets()

    pattern = Pattern(
        ModalityLoadInfo(Landmarks, input_length, (input_length, 136)),
        ModalityLoadInfo(Landmarks, output_length, (output_length // frame_per_prediction,
                                                    136 * frame_per_prediction))
    )
    anomaly_pattern = Pattern(
        *pattern,
        "labels"
    )

    train_dataset = train_subset.make_tf_dataset(pattern.with_added_depth())
    train_dataset = train_dataset.map(add_batch_noise)
    test_dataset = test_subset.make_tf_dataset(pattern.with_added_depth())
    # endregion

    # tmp = Model(inputs=landmarks_transformer.inputs,
    #             outputs=landmarks_transformer.decoder_attention_weights)
    # self_pwet, encoder_pwet = tmp.predict(train_dataset.batch(1), steps=1)
    # self_pwet = np.squeeze(self_pwet)
    # encoder_pwet = np.squeeze(encoder_pwet)
    # layer_count, head_count, _, _ = self_pwet.shape
    # import matplotlib.pyplot as plt
    # for i in range(layer_count):
    #     for j in range(head_count):
    #         plt.pcolormesh(self_pwet[i, j], cmap="jet")
    #         plt.show()
    #         plt.pcolormesh(encoder_pwet[i, j], cmap="jet")
    #         plt.show()
    # exit()

    # region Callbacks
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(landmarks_transformer, log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    # region Landmarks (video)
    # region Default
    lvc_train_dataset = train_dataset.map(lambda data: (data, data[1]))
    lvc_test_dataset = test_dataset.map(lambda data: (data, data[1]))

    train_landmarks_callback = LandmarksVideoCallback(lvc_train_dataset, landmarks_transformer, tensorboard,
                                                      is_train_callback=True, fps=25)
    test_landmarks_callback = LandmarksVideoCallback(lvc_test_dataset, landmarks_transformer, tensorboard,
                                                     is_train_callback=False)
    # endregion

    # region Autonomous
    lvc_train_dataset = train_dataset.map(lambda data: data)
    lvc_test_dataset = test_dataset.map(lambda data: data)
    autonomous_train_landmarks_callback = LandmarksVideoCallback(lvc_train_dataset, autonomous_transformer,
                                                                 tensorboard, is_train_callback=True,
                                                                 prefix="autonomous")
    autonomous_test_landmarks_callback = LandmarksVideoCallback(lvc_test_dataset, autonomous_transformer,
                                                                tensorboard, is_train_callback=False,
                                                                prefix="autonomous")
    # endregion
    # endregion

    # region AUC
    # region Default (32 frames)
    # raw_predictions = RawPredictionsLayer(output_length=input_length)([landmarks_transformer.output,
    #                                                                    landmarks_transformer.decoder_target_layer])
    # raw_predictions_model = Model(inputs=landmarks_transformer.inputs,
    #                               outputs=raw_predictions,
    #                               name="raw_predictions_model_{}".format(input_length))
    #
    # auc_callback_32 = make_auc_callback(test_subset=test_subset,
    #                                     pattern=anomaly_pattern,
    #                                     predictions_model=raw_predictions_model,
    #                                     tensorboard=tensorboard,
    #                                     samples_count=512,
    #                                     prefix=str(input_length))
    # endregion
    # region Default (128 frames)
    raw_predictions = RawPredictionsLayer(output_length=output_length)([landmarks_transformer.output,
                                                                        landmarks_transformer.decoder_target_layer])
    raw_predictions_model = Model(inputs=landmarks_transformer.inputs, outputs=raw_predictions,
                                  name="raw_predictions_model_{}".format(output_length))

    auc_callback_128 = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                               test_subset=test_subset, pattern=anomaly_pattern, samples_count=512,
                                               prefix=str(output_length))
    # endregion
    # region Autonomous
    autonomous_ground_truth = Input(batch_shape=autonomous_transformer.output.shape, name="autonomous_ground_truth")
    autonomous_raw_predictions = RawPredictionsLayer()([autonomous_transformer.output, autonomous_ground_truth])
    autonomous_raw_predictions_model = Model(inputs=[autonomous_transformer.input, autonomous_ground_truth],
                                             outputs=autonomous_raw_predictions,
                                             name="autonomous_raw_predictions_model")
    autonomous_auc_callback = AUCCallback.from_subset(predictions_model=autonomous_raw_predictions_model,
                                                      tensorboard=tensorboard, test_subset=test_subset,
                                                      pattern=anomaly_pattern, samples_count=512, prefix="autonomous")
    # endregion
    # endregion

    model_checkpoint = TmpModelCheckpoint(os.path.join(base_log_dir, "weights_{epoch:03d}.hdf5"),
                                          verbose=1)
    callbacks = [model_checkpoint,
                 tensorboard,
                 train_landmarks_callback,
                 test_landmarks_callback,
                 autonomous_train_landmarks_callback,
                 autonomous_test_landmarks_callback,
                 # auc_callback_32,
                 auc_callback_128,
                 autonomous_auc_callback
                 ]

    # endregion

    train_dataset = train_dataset.batch(batch_size).prefetch(-1)
    test_dataset = test_dataset.batch(batch_size)

    # autonomous_train_dataset = train_dataset.map(lambda x: x)
    # autonomous_test_dataset = test_dataset.map(lambda x: x)

    landmarks_transformer.fit(train_dataset, epochs=50, steps_per_epoch=25000,
                              validation_data=test_dataset, validation_steps=2000,
                              callbacks=callbacks)

    # autonomous_transformer.fit(autonomous_train_dataset, epochs=30, steps_per_epoch=2000,
    #                            validation_data=autonomous_test_dataset, validation_steps=500,
    #                            callbacks=callbacks)

    anomaly_detector = AnomalyDetector(inputs=landmarks_transformer.inputs,
                                       output=landmarks_transformer.output,
                                       ground_truth=landmarks_transformer.inputs[1])
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=anomaly_pattern.with_added_depth(),
                                          log_dir=log_dir,
                                          stride=1,
                                          pre_normalize_predictions=True)


# endregion
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
        ModalityLoadInfo(MelSpectrogram, input_length, (input_length, n_mel_filters)),
        ModalityLoadInfo(MelSpectrogram, output_length, (output_length, n_mel_filters))
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
    log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/mfcc"
    log_dir = os.path.join(log_dir, "log_{}".format(int(time())))
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

    raw_predictions = RawPredictionsLayer()([audio_autoencoder.output, audio_autoencoder.input])
    raw_predictions_model = Model(inputs=audio_autoencoder.input, outputs=raw_predictions,
                                  name="raw_predictions_model")
    auc_callback = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=test_subset, pattern=anomaly_pattern)

    # model_checkpoint = ModelCheckpoint("../logs/tests/kitchen_sink/mfcc_only/weights.{epoch:03d}.hdf5", )
    model_checkpoint = TmpModelCheckpoint(os.path.join(log_dir, "weights_{epoch:03d}.hdf5"))
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

    anomaly_detector = AnomalyDetector(inputs=audio_autoencoder.inputs,
                                       output=audio_autoencoder.output,
                                       ground_truth=audio_autoencoder.inputs)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=anomaly_pattern,
                                          log_dir=log_dir,
                                          stride=1)


# endregion

def main():
    train_scaling_video_transformer()


if __name__ == "__main__":
    main()
