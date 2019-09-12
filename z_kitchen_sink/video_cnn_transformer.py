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
from anomaly_detection import RawPredictionsModel, AnomalyDetector, known_metrics
from callbacks import EagerModelCheckpoint
from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols import ImageCallbackConfig, AUCCallbackConfig


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
                              layers_intermediate_size=code_size // 4,
                              layers_count=4,
                              attention_key_size=code_size // 4,
                              attention_heads_count=4,
                              attention_values_size=code_size // 4,
                              dropout_rate=0.1,
                              positional_encoding_range=0.1,
                              mode=TransformerMode.CONTINUOUS,
                              copy_regularization_factor=0.0,
                              name="Transformer")

    cnn_transformer = CNNTransformer(input_length=input_length,
                                     output_length=output_length,
                                     autoencoder=ae,
                                     transformer=transformer,
                                     learning_rate=1e-4,
                                     train_only_embeddings=False,
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


def get_dataset_folder(dataset_name) -> str:
    if dataset_name is "ped2":
        return "../datasets/ucsd/ped2"
    elif dataset_name is "ped1":
        return "../datasets/ucsd/ped1"
    elif dataset_name is "emoly":
        return "../datasets/emoly"
    else:
        raise ValueError(dataset_name)


def make_augment_video_function(video_length, height, width, channels):
    def augment_video(video: tf.Tensor):
        crop_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.0)
        original_shape = tf.cast(tf.shape(video), tf.float32)
        original_height, original_width = original_shape[1], original_shape[2]
        crop_size = [video_length, crop_ratio * original_height, crop_ratio * original_width, channels]
        video = tf.image.random_crop(video, crop_size)
        # video = tf.image.random_brightness(video, max_delta=0.05)
        video = tf.image.resize(video, (height, width))

        return video

    return augment_video


def get_video_length(model: Model, input_length: int, output_length: int, time_step: int) -> int:
    if isinstance(model, CNNTransformer):
        video_length = time_step * (input_length + output_length)
    elif isinstance(model, AEP):
        video_length = time_step * 2
    elif isinstance(model, IAE):
        video_length = time_step * input_length
    else:
        video_length = time_step
    return video_length


def make_preprocess_video_function(height, width):
    def preprocess_video(video: tf.Tensor):
        video = tf.image.resize(video, (height, width))
        return video

    return preprocess_video


def train_video_cnn_transformer(input_length=4,
                                output_length=4,
                                time_step=4,
                                height=128,
                                width=128,
                                channels=1,
                                code_size=256,
                                initial_epoch=0,
                                use_transformer=False,
                                batch_size=8,
                                dataset_name="ped2"):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 mode="IAE")
    model = transformer if use_transformer else ae

    video_length = get_video_length(model, input_length, output_length, time_step)

    # region Pattern
    augment_video = make_augment_video_function(video_length, height, width, channels)
    preprocess_video = make_preprocess_video_function(height, width)

    video_shape = (video_length, height, width, channels)
    train_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, augment_video)
    )
    test_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video)
    )
    anomaly_pattern = Pattern(*test_pattern, "labels")
    # endregion

    protocol = Protocol(model=model,
                        dataset_name=dataset_name,
                        protocol_name="video_cnn_transformer")

    # region Image callbacks configs
    image_callbacks_configs = [
        ImageCallbackConfig(model, test_pattern, True, "train"),
        ImageCallbackConfig(model, test_pattern, False, "test"),
    ]
    if use_transformer:
        image_callbacks_configs += [ImageCallbackConfig(transformer.evaluator, test_pattern, False, "evaluator_test")]
    elif isinstance(ae, IAE):
        image_callbacks_configs += [ImageCallbackConfig(ae.interpolate, test_pattern, False, "interpolate_test")]
    # endregion

    # region AUC callbacks configs
    if use_transformer:
        auc_callbacks_configs = [
            AUCCallbackConfig(transformer, anomaly_pattern, output_length, prefix="trainer"),
            AUCCallbackConfig(transformer.evaluator, anomaly_pattern, output_length, prefix="evaluator"),
        ]
    else:
        auc_callbacks_configs = [
            AUCCallbackConfig(model, anomaly_pattern, video_length, prefix="")
        ]
        if isinstance(ae, IAE):
            auc_callbacks_configs += [AUCCallbackConfig(model.interpolate, anomaly_pattern, video_length, prefix="iae")]
    # endregion

    config = ProtocolTrainConfig(batch_size=batch_size,
                                 pattern=train_pattern,
                                 epochs=100,
                                 initial_epoch=initial_epoch,
                                 image_callbacks_configs=image_callbacks_configs,
                                 auc_callbacks_configs=auc_callbacks_configs,
                                 early_stopping_metric=model.metrics_names[0])

    protocol.train_model(config=config)


def _train_video_cnn_transformer(input_length=4,
                                 output_length=4,
                                 time_step=4,
                                 height=128,
                                 width=128,
                                 channels=1,
                                 code_size=256,
                                 initial_epoch=0,
                                 use_transformer=False,
                                 batch_size=8,
                                 dataset_name="ped2"):
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
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_cnn_transformer/{}".format(dataset_name)
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(model, log_dir)
    weights_path = os.path.join(log_dir, "weights_{epoch:03d}.hdf5")
    # endregion

    if initial_epoch > 0:
        model.load_weights(os.path.join(base_log_dir, "weights_{:03d}.hdf5").format(initial_epoch))

    video_length = get_video_length(model, input_length, output_length, time_step)

    # region Dataset
    augment_video = make_augment_video_function(video_length, height, width, channels)
    preprocess_video = make_preprocess_video_function(height, width)

    video_shape = (video_length, height, width, channels)
    train_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, augment_video)
    )
    test_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video)
    )
    anomaly_pattern = Pattern(*test_pattern, "labels")

    dataset_config = DatasetConfig(get_dataset_folder(dataset_name), output_range=(0.0, 1.0))
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
    model_checkpoint = EagerModelCheckpoint(weights_path)
    callbacks.append(model_checkpoint)
    # endregion
    # region Early stopping
    early_stopping = EarlyStopping(monitor=model.metrics_names[0], mode="min", restore_best_weights=True,
                                   patience=3)
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
                               use_transformer=False,
                               dataset_name="ped2"):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 mode="IAE")
    model = transformer if use_transformer else ae

    video_length = get_video_length(model, input_length, output_length, time_step)

    # region Pattern
    preprocess_video = make_preprocess_video_function(height, width)

    video_shape = (video_length, height, width, channels)
    pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video),
        "labels"
    )
    # endregion

    use_default_model = True
    model_used = model.name
    autoencoder = model
    if not use_default_model:
        if use_transformer:
            autoencoder = transformer.evaluator
            model_used = "transformer.evaluator"
        elif isinstance(ae, IAE):
            autoencoder = ae.interpolate
            model_used = "ae.interpolate"

    protocol = Protocol(model=model,
                        dataset_name=dataset_name,
                        protocol_name="video_cnn_transformer",
                        autoencoder=autoencoder,
                        model_name=model_used,
                        )

    total_output_length = output_length * time_step if use_transformer else video_length
    config = ProtocolTestConfig(pattern=pattern,
                                epoch=initial_epoch,
                                output_length=total_output_length,
                                detector_stride=1,
                                pre_normalize_predictions=True,
                                )

    protocol.test_model(config=config)


def _test_video_cnn_transformer(input_length=4,
                                output_length=4,
                                time_step=4,
                                height=128,
                                width=128,
                                channels=1,
                                code_size=256,
                                initial_epoch=15,
                                use_transformer=False,
                                dataset_name="ped2"):
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
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_cnn_transformer/{}".format(dataset_name)
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

    preprocess_video = make_preprocess_video_function(height, width)

    video_shape = (video_length, height, width, channels)
    anomaly_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length, video_shape, preprocess_video),
        "labels"
    )

    dataset_config = DatasetConfig(get_dataset_folder(dataset_name), output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    detector_metrics = list(known_metrics.keys())
    detector_stride = 1
    pre_normalize_predictions = True
    use_default_model = True

    if not use_default_model:
        if use_transformer:
            model = transformer.evaluator
            model_used = "transformer.evaluator"
        elif isinstance(ae, IAE):
            model = ae.interpolate
            model_used = "ae.interpolate"
        else:
            model_used = "default"
    else:
        model_used = "default"

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
                                              "model_used": model_used,
                                          }
                                          )
