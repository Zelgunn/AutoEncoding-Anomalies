import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Flatten, Dense, Add
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
import os
from time import time
from typing import Tuple

from callbacks import ImageCallback, AUCCallback
from datasets.loaders import DatasetConfig, DatasetLoader
from modalities import ModalityLoadInfo, RawVideo, Pattern
from utils.train_utils import save_model_info
from transformers import Transformer
from transformers.Transformer import TransformerMode
from models import AE, AEP, VAE, CNNTransformer, IAE
from anomaly_detection import RawPredictionsModel, AnomalyDetector
from z_kitchen_sink.utils import TmpModelCheckpoint


def make_encoder(input_layer, code_size: int, common_cnn_params, name="Encoder") -> Model:
    x = input_layer
    x = Conv3D(filters=96, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=128, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=256, strides=(2, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=256, strides=(2, 2, 2), kernel_size=(2, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    y = Conv3D(filters=256, strides=1, padding="same", kernel_size=(1, 3, 3), activation="relu")(x)
    x = Add()([x, y])
    x = Conv3D(filters=code_size, strides=1, kernel_size=(1, 3, 3), padding="same")(x)

    encoder_output = x
    encoder = Model(inputs=input_layer, outputs=encoder_output, name=name)
    return encoder


def make_decoder(input_layer, channels: int, common_cnn_params, name="Decoder") -> Model:
    x = input_layer

    x = Conv3DTranspose(filters=256, strides=(2, 2, 2), kernel_size=(1, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=256, strides=(2, 2, 2), kernel_size=(2, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=128, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=96, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=channels, strides=1, kernel_size=3, padding="same", activation="sigmoid")(x)

    decoder_output = x
    decoder = Model(inputs=input_layer, outputs=decoder_output, name=name)
    return decoder


def make_discriminator(input_layer, code_size: int, common_cnn_params, name="Discriminator"):
    x = input_layer

    x = Conv3D(filters=16, strides=(1, 2, 2), **common_cnn_params)(x)
    x = Conv3D(filters=32, strides=(1, 2, 2), **common_cnn_params)(x)
    x = Conv3D(filters=32, strides=2, **common_cnn_params)(x)
    x = Conv3D(filters=64, strides=2, **common_cnn_params)(x)
    x = Flatten()(x)
    x = Dense(units=code_size // 2, activation="relu", kernel_initializer="glorot_uniform")(x)
    discriminator_intermediate_output = x

    discriminator_output = Dense(units=1, activation=None)(x)

    discriminator = Model(inputs=input_layer, outputs=[discriminator_output, discriminator_intermediate_output],
                          name=name)
    return discriminator


def make_video_cnn_transformer(input_length: int,
                               output_length: int,
                               time_step: int,
                               height: int,
                               width: int,
                               channels: int,
                               code_size: int,
                               mode="AE",
                               ) -> Tuple[Model, CNNTransformer]:
    video_shape = (time_step, height, width, channels)
    common_cnn_params = {
        "padding": "same",
        "activation": "relu",
        "kernel_initializer": "glorot_uniform",
        "use_bias": True
    }

    input_layer = Input(video_shape, name="InputLayer")
    decoder_input_layer = Input(shape=[1, 8, 8, code_size], name="DecoderInputLayer")

    encoder_code_size = code_size if mode != "VAE" else code_size * 2
    encoder = make_encoder(input_layer, encoder_code_size, common_cnn_params)
    decoder = make_decoder(decoder_input_layer, channels, common_cnn_params)

    # region VAEGAN

    # vaegan = VAEGAN(encoder=encoder,
    #                 decoder=decoder,
    #                 discriminator=discriminator,
    #                 reconstruction_loss_factor=1000.0,
    #                 learned_reconstruction_loss_factor=10.0,
    #                 kl_divergence_loss_factor=0.1,
    #                 autoencoder_learning_rate=1e-3,
    #                 discriminator_learning_rate=1e-4,
    #                 balance_discriminator_learning_rate=True,
    #                 loss_used=1,
    #                 name="VAEGAN")

    # endregion

    # region AE
    if mode == "AE":
        ae = AE(encoder=encoder,
                decoder=decoder,
                learning_rate=1e-3)
    elif mode == "VAE":
        ae = VAE(encoder=encoder,
                 decoder=decoder,
                 learning_rate=1e-3,
                 reconstruction_loss_factor=1e3,
                 kl_divergence_factor=2e-1)
    elif mode == "AEP":
        predictor = make_decoder(decoder_input_layer, channels, common_cnn_params, name="Predictor")
        ae = AEP(encoder=encoder,
                 decoder=decoder,
                 predictor=predictor,
                 input_length=time_step,
                 learning_rate=1e-3)
    elif mode == "IAE":
        ae = IAE(encoder=encoder,
                 decoder=decoder,
                 step_size=time_step,
                 learning_rate=1e-3)
    else:
        raise NotImplementedError

    # endregion

    # region CNN transformer

    # def sample_encoder(inputs):
    #     mean, variance = tf.split(inputs, num_or_size_splits=2, axis=-1)
    #     distribution = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=variance)
    #     return distribution.sample()

    transformer = Transformer(input_size=code_size,
                              output_size=code_size,
                              output_activation="linear",
                              layers_intermediate_size=code_size // 2,
                              layers_count=4,
                              attention_key_size=code_size // 2,
                              attention_heads_count=4,
                              attention_values_size=code_size // 2,
                              dropout_rate=0.0,
                              positional_encoding_range=0.1,
                              mode=TransformerMode.CONTINUOUS,
                              copy_regularization_factor=0.0,
                              name="Transformer")

    cnn_transformer = CNNTransformer(input_length=input_length,
                                     output_length=output_length,
                                     autoencoder=ae,
                                     transformer=transformer,
                                     learning_rate=1e-4,
                                     name="CNNTransformer")

    # endregion

    return ae, cnn_transformer


def make_ae_auc_callback(autoencoder: AE,
                         tensorboard: TensorBoard,
                         dataset_loader: DatasetLoader,
                         pattern: Pattern,
                         output_length: int,
                         prefix="",
                         ) -> AUCCallback:
    raw_predictions_model = RawPredictionsModel(autoencoder,
                                                output_length=output_length,
                                                name="AutoencoderRawPredictionsModel")

    return AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                   test_subset=dataset_loader.test_subset, pattern=pattern,
                                   samples_count=128, epoch_freq=1, batch_size=4, prefix=prefix)


def make_transformer_auc_callback(transformer: CNNTransformer,
                                  tensorboard: TensorBoard,
                                  dataset_loader: DatasetLoader,
                                  pattern: Pattern,
                                  output_length: int,
                                  auc_mode: str,
                                  ) -> AUCCallback:
    if auc_mode is "Evaluator":
        model_used = transformer.evaluator
    else:
        model_used = transformer

    raw_predictions_model = RawPredictionsModel(model_used,
                                                output_length=output_length,
                                                name="AutoencoderRawPredictionsModel")

    return AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                   test_subset=dataset_loader.test_subset, pattern=pattern,
                                   samples_count=128, epoch_freq=1, batch_size=4, prefix=auc_mode)


def train_video_cnn_transformer(input_length=4,
                                output_length=4,
                                time_step=4,
                                height=128,
                                width=128,
                                channels=1,
                                code_size=256,
                                initial_epoch=0,
                                use_transformer=False,
                                batch_size=8):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 mode="IAE")

    model = transformer if use_transformer else ae

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_cnn_transformer"
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(model, log_dir)
    weights_path = os.path.join(log_dir, "weights_{epoch:03d}.hdf5")
    # endregion

    if initial_epoch > 0:
        ae.load_weights(os.path.join(base_log_dir, "weights_{:03d}.hdf5").format(initial_epoch))

    ae.trainable = model is ae
    # region Video length
    if model is transformer:
        video_length = time_step * (input_length + output_length)
    elif isinstance(ae, AEP):
        video_length = time_step * 2
    elif isinstance(ae, IAE):
        video_length = time_step * input_length
    else:
        video_length = time_step

    # endregion

    # tmp = transformer(tf.random.normal([16, 8, 4, 128, 128, 1]))

    # region Dataset
    def augment_video(video: tf.Tensor):
        crop_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.0)
        original_shape = tf.cast(tf.shape(video), tf.float32)
        original_height, original_width = original_shape[1], original_shape[2]
        crop_size = [video_length, crop_ratio * original_height, crop_ratio * original_width, channels]
        video = tf.image.random_crop(video, crop_size)
        # video = tf.image.random_brightness(video, max_delta=0.05)
        video = preprocess_video(video)

        return video

    def preprocess_video(video: tf.Tensor):
        video = tf.image.resize(video, (height, width))
        return video

    video_shape = (video_length, height, width, channels)
    train_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, augment_video)
    )
    test_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video)
    )
    anomaly_pattern = Pattern(*test_pattern, "labels")

    dataset_config = DatasetConfig("../datasets/ucsd/ped2", output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    dataset = dataset_loader.train_subset.make_tf_dataset(train_pattern)
    dataset = dataset.batch(batch_size).prefetch(-1)

    val_dataset = dataset_loader.test_subset.make_tf_dataset(train_pattern)
    val_dataset = val_dataset.batch(batch_size)
    # endregion

    # region Callbacks
    tensorboard = TensorBoard(log_dir=log_dir, update_freq=16, profile_batch=0)
    callbacks = [tensorboard]
    # region Image Callbacks
    train_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=model,
                                                                subset=dataset_loader.train_subset,
                                                                pattern=test_pattern,
                                                                name="train",
                                                                is_train_callback=True,
                                                                tensorboard=tensorboard,
                                                                epoch_freq=1)
    test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=model,
                                                               subset=dataset_loader.test_subset,
                                                               pattern=test_pattern,
                                                               name="test",
                                                               is_train_callback=False,
                                                               tensorboard=tensorboard,
                                                               epoch_freq=1)
    callbacks += train_image_callbacks
    callbacks += test_image_callbacks

    if model is transformer:
        evaluator_image_callbacks = ImageCallback.from_model_and_subset(transformer.evaluator,
                                                                        subset=dataset_loader.test_subset,
                                                                        pattern=test_pattern,
                                                                        name="evaluator_test",
                                                                        is_train_callback=False,
                                                                        tensorboard=tensorboard,
                                                                        epoch_freq=1
                                                                        )
        callbacks += evaluator_image_callbacks
    # endregion
    # region Checkpoint
    model_checkpoint = TmpModelCheckpoint(weights_path)
    callbacks.append(model_checkpoint)
    # endregion
    # region Early stopping
    early_stopping = EarlyStopping(monitor=model.metrics_names[0], mode="min", restore_best_weights=True,
                                   patience=1)
    callbacks.append(early_stopping)
    # endregion
    # region AUC
    if model is transformer:
        trainer_auc_callback = make_transformer_auc_callback(transformer, tensorboard, dataset_loader, anomaly_pattern,
                                                             output_length, auc_mode="Trainer")
        callbacks.append(trainer_auc_callback)
        evaluator_auc_callback = make_transformer_auc_callback(transformer, tensorboard, dataset_loader,
                                                               anomaly_pattern, output_length, auc_mode="Evaluator")
        callbacks.append(evaluator_auc_callback)
    else:
        auc_callback = make_ae_auc_callback(ae, tensorboard, dataset_loader, anomaly_pattern,
                                            output_length=video_length)
        callbacks.append(auc_callback)

        if isinstance(ae, IAE):
            iae_auc_callback = make_ae_auc_callback(ae.interpolate, tensorboard, dataset_loader, anomaly_pattern,
                                                    output_length=video_length, prefix="iae")
            callbacks.append(iae_auc_callback)
    # endregion
    # endregion

    model.fit(dataset, steps_per_epoch=1000, epochs=100,
              validation_data=val_dataset, validation_steps=100,
              callbacks=callbacks, initial_epoch=initial_epoch)


def test_video_cnn_transformer(input_length=4,
                               output_length=4,
                               time_step=4,
                               height=128,
                               width=128,
                               channels=1,
                               code_size=256,
                               initial_epoch=15,
                               use_transformer=False):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 mode="IAE")

    model = transformer if use_transformer else ae

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_cnn_transformer"
    log_dir = os.path.join(base_log_dir, "test_log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(model, log_dir)
    # endregion

    if initial_epoch > 0:
        model.load_weights(os.path.join(base_log_dir, "weights_{:03d}.hdf5").format(initial_epoch))

    # region Video length
    if model is transformer:
        video_length = time_step * (input_length + output_length)
    elif isinstance(ae, AEP):
        video_length = time_step * 2
    elif isinstance(ae, IAE):
        video_length = time_step * input_length
    else:
        video_length = time_step

    # endregion

    def preprocess_video(video: tf.Tensor):
        video = tf.image.resize(video, (height, width))
        return video

    video_shape = (video_length, height, width, channels)
    anomaly_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video),
        "labels"
    )

    dataset_config = DatasetConfig("../datasets/ucsd/ped2", output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    detector_metrics = ["mse", "ssim", "mae"]
    detector_stride = 2
    pre_normalize_predictions = True
    use_evaluator = False

    if model is transformer and use_evaluator:
        model = transformer.evaluator

    total_output_length = output_length * time_step if use_transformer else video_length
    anomaly_detector = AnomalyDetector(autoencoder=model,
                                       output_length=total_output_length,
                                       metrics=detector_metrics)

    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=anomaly_pattern.with_added_depth().with_added_depth(),
                                          log_dir=log_dir,
                                          stride=detector_stride,
                                          pre_normalize_predictions=pre_normalize_predictions,
                                          additional_config={
                                              "initial_epoch": initial_epoch,
                                              "use_evaluator": use_evaluator,
                                          }
                                          )
