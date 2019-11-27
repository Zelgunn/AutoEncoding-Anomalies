import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Flatten, Dense, Add
from tensorflow.python.keras.layers import LeakyReLU
from typing import Tuple

from modalities import ModalityLoadInfo, RawVideo, Pattern
from transformers import Transformer
from transformers.Transformer import TransformerMode
from models import AE, AEP, VAE, CNNTransformer, IAE
from protocols import Protocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols import ImageCallbackConfig, AUCCallbackConfig
from CustomKerasLayers import ResBlock3D, ResBlock3DTranspose


def make_encoder(input_layer, code_size: int, common_cnn_params, name="Encoder") -> Model:
    x = input_layer
    x = Conv3D(filters=32, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=64, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=128, strides=(2, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=128, strides=(2, 2, 2), kernel_size=(2, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    y = Conv3D(filters=128, strides=1, padding="same", kernel_size=(1, 3, 3), activation="relu")(x)
    x = Add()([x, y])
    x = Conv3D(filters=code_size, strides=1, kernel_size=(1, 3, 3), padding="same")(x)

    encoder_output = x
    encoder = Model(inputs=input_layer, outputs=encoder_output, name=name)
    return encoder


def make_decoder(input_layer, channels: int, common_cnn_params, name="Decoder") -> Model:
    x = input_layer

    x = Conv3DTranspose(filters=128, strides=(2, 2, 2), kernel_size=(1, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=128, strides=(2, 2, 2), kernel_size=(2, 3, 3), **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=64, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=32, strides=(1, 2, 2), kernel_size=3, **common_cnn_params)(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=channels, strides=1, kernel_size=3, padding="same", activation="sigmoid")(x)

    decoder_output = x
    decoder = Model(inputs=input_layer, outputs=decoder_output, name=name)
    return decoder


def make_residual_encoder(input_layer, code_size: int, name="ResidualEncoder") -> Model:
    leaky_relu = LeakyReLU(alpha=1e-2)

    x = input_layer
    x = ResBlock3D(filters=8, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3D(filters=12, strides=(2, 1, 1), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3D(filters=18, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3D(filters=27, strides=(2, 1, 1), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3D(filters=40, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3D(filters=code_size, strides=1, kernel_size=3, activation="sigmoid")(x)

    encoder_output = x
    encoder = Model(inputs=input_layer, outputs=encoder_output, name=name)
    return encoder


def make_residual_decoder(input_layer, channels: int, name="ResidualDecoder") -> Model:
    leaky_relu = LeakyReLU(alpha=1e-2)

    x = input_layer
    x = ResBlock3DTranspose(filters=64, strides=1, kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3DTranspose(filters=40, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3DTranspose(filters=27, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3DTranspose(filters=18, strides=(2, 1, 1), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3DTranspose(filters=12, strides=(1, 2, 2), kernel_size=3, activation=leaky_relu)(x)
    x = ResBlock3DTranspose(filters=8, strides=(2, 1, 1), kernel_size=3, activation=leaky_relu)(x)
    x = Conv3D(filters=channels, strides=1, kernel_size=1, padding="same", activation="linear", use_bias=False)(x)

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
                               train_only_embeddings: bool,
                               copy_regularization_factor: float,
                               mode="AE",
                               use_residual=False,
                               ) -> Tuple[Model, CNNTransformer]:
    mode = mode.upper()
    video_shape = (time_step, height, width, channels)
    common_cnn_params = {
        "padding": "same",
        "activation": "relu",
        "kernel_initializer": "glorot_uniform",
        "use_bias": True
    }

    input_layer = Input(video_shape, name="InputLayer")
    decoder_input_layer = Input(shape=[time_step // 4, height // 8, width // 8, code_size], name="DecoderInputLayer")

    encoder_code_size = code_size if mode != "VAE" else code_size * 2

    if use_residual:
        encoder = make_residual_encoder(input_layer, encoder_code_size)
        decoder = make_residual_decoder(decoder_input_layer, channels)
    else:
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

    transformer = Transformer(input_size=code_size,
                              output_size=code_size,
                              output_activation="sigmoid",
                              layers_intermediate_size=code_size,
                              layers_count=4,
                              attention_key_size=code_size // 4,
                              attention_heads_count=4,
                              attention_values_size=code_size // 4,
                              dropout_rate=0.0,
                              positional_encoding_range=0.1,
                              mode=TransformerMode.CONTINUOUS,
                              copy_regularization_factor=copy_regularization_factor,
                              name="Transformer")

    cnn_transformer = CNNTransformer(input_length=input_length,
                                     output_length=output_length,
                                     autoencoder=ae,
                                     transformer=transformer,
                                     learning_rate=1e-4,
                                     train_only_embeddings=train_only_embeddings,
                                     name="CNNTransformer")

    # endregion

    return ae, cnn_transformer


def make_augment_video_function(video_length: int,
                                height: int,
                                width: int,
                                channels: int,
                                to_grayscale: bool,
                                extract_face: bool
                                ):
    preprocess_video = make_preprocess_video_function(height, width, to_grayscale, extract_face)

    def augment_video(video: tf.Tensor,
                      bounding_boxes: tf.Tensor = None,
                      ) -> tf.Tensor:
        if extract_face:
            video = preprocess_video(video, bounding_boxes)

        crop_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.0)
        original_shape = tf.cast(tf.shape(video), tf.float32)
        original_height, original_width = original_shape[1], original_shape[2]
        crop_size = [video_length, crop_ratio * original_height, crop_ratio * original_width, channels]
        video = tf.image.random_crop(video, crop_size)

        if not extract_face:
            video = preprocess_video(video)

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


def make_preprocess_video_function(height: int,
                                   width: int,
                                   to_grayscale: bool,
                                   extract_face: bool,
                                   ):
    def resize_video(video: tf.Tensor,
                     labels: tf.Tensor = None):
        video = tf.image.resize(video, (height, width))

        if to_grayscale:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
            video *= rgb_weights
            video = tf.reduce_sum(video, axis=-1, keepdims=True)

        if labels is not None:
            return video, labels
        return video

    def full_preprocess(video: tf.Tensor,
                        bounding_boxes: tf.Tensor,
                        labels: tf.Tensor = None
                        ):
        video_shape = tf.shape(video)
        source_height = tf.cast(video_shape[1], tf.float32)
        source_width = tf.cast(video_shape[2], tf.float32)

        boxes = bounding_boxes[0]

        start_y, end_y = boxes[0] * source_height, boxes[1] * source_height
        start_x, end_x = boxes[2] * source_width, boxes[3] * source_width

        start_y, end_y = tf.cast(start_y, tf.int32), tf.cast(end_y, tf.int32)
        start_x, end_x = tf.cast(start_x, tf.int32), tf.cast(end_x, tf.int32)

        video = video[:, start_y:end_y, start_x:end_x]

        tf.assert_equal(tf.reduce_any(tf.math.is_nan(boxes)), False)

        return resize_video(video, labels)

    preprocess_video = full_preprocess if extract_face else resize_video

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
                                autoencoder_mode="iae",
                                use_residual=False,
                                train_only_embeddings=True,
                                copy_regularization_factor=1.0,
                                batch_size=8,
                                dataset_name="ped2"):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 train_only_embeddings=train_only_embeddings,
                                                 copy_regularization_factor=copy_regularization_factor,
                                                 mode=autoencoder_mode,
                                                 use_residual=use_residual)
    model = transformer if use_transformer else ae

    video_length = get_video_length(model, input_length, output_length, time_step)

    # region Pattern
    extract_face = dataset_name in ["emoly"]
    to_grayscale = dataset_name not in ["ucsd"]
    preprocess_video = make_preprocess_video_function(height, width, to_grayscale, extract_face)
    augment_video = make_augment_video_function(video_length, height, width, channels, to_grayscale, extract_face)
    # augment_video = preprocess_video

    train_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length),
        # ModalityLoadInfo(Faces, video_length),
        output_map=augment_video
    )
    test_pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length),
        # ModalityLoadInfo(Faces, video_length),
        output_map=preprocess_video
    )
    anomaly_pattern = Pattern(*test_pattern, "labels", output_map=preprocess_video)
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
                                 early_stopping_metric=model.metrics_names[0],
                                 steps_per_epoch=1000,
                                 validation_steps=128)

    try:
        protocol.train_model(config=config)
    except KeyboardInterrupt:
        pass

    total_output_length = output_length * time_step if use_transformer else video_length
    test_config = ProtocolTestConfig(pattern=anomaly_pattern,
                                     epoch=initial_epoch,
                                     output_length=total_output_length,
                                     detector_stride=total_output_length,
                                     pre_normalize_predictions=True,
                                     )

    protocol.test_model(config=test_config)


def test_video_cnn_transformer(input_length=4,
                               output_length=4,
                               time_step=4,
                               height=128,
                               width=128,
                               channels=1,
                               code_size=256,
                               initial_epoch=15,
                               use_transformer=False,
                               autoencoder_mode="iae",
                               use_residual=False,
                               dataset_name="ped2"):
    ae, transformer = make_video_cnn_transformer(input_length=input_length,
                                                 output_length=output_length,
                                                 time_step=time_step,
                                                 height=height,
                                                 width=width,
                                                 channels=channels,
                                                 code_size=code_size,
                                                 train_only_embeddings=False,
                                                 copy_regularization_factor=1.0,
                                                 mode=autoencoder_mode,
                                                 use_residual=use_residual)
    model = transformer if use_transformer else ae

    video_length = get_video_length(model, input_length, output_length, time_step)

    # region Pattern
    extract_face = dataset_name in ["emoly"]
    to_grayscale = dataset_name not in ["ucsd"]
    preprocess_video = make_preprocess_video_function(height, width, to_grayscale, extract_face)

    pattern = Pattern(
        ModalityLoadInfo(RawVideo, video_length),
        # ModalityLoadInfo(Faces, video_length),
        "labels",
        output_map=preprocess_video
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
