import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Reshape
from typing import List

from custom_tf_models import AE
from custom_tf_models.adversarial import CoupledVAEGANs
from protocols.utils import get_encoder_layers, get_decoder_layers, make_discriminator, to_sequential
from modalities import Pattern, ModalityLoadInfo, RawVideo, RawAudio, Faces
from data_processing.video_processing.video_preprocessing import make_video_preprocessor
from protocols import Protocol, ProtocolTrainConfig
from callbacks.configs import ImageCallbackConfig, AUCCallbackConfig


def compute_output_shape(layers: List[Layer], input_shape):
    input_shape = (None, *input_shape)
    for layer in layers:
        input_shape = layer.compute_output_shape(input_shape)
    return input_shape[1:]


def main():
    initial_epoch = 0

    # region Constants
    video_length = 25
    video_height = video_width = 32
    video_channels = 3

    audio_length_multiplier = 1920  # for wave
    # audio_length_multiplier = 4  # for mfcc
    audio_length = video_length * audio_length_multiplier
    audio_channels = 2  # for wave
    # audio_channels = 100  # for mfcc
    # endregion

    # region Hyper-parameters
    batch_size = 4
    steps_per_epoch = 1000
    epochs = 100
    validation_steps = 64
    seed = 42
    code_size = 128
    code_length = video_length

    # region Video
    video_encoder_filters = [8, 16, 32, 64, 128]
    video_encoder_kernel_size = 3
    video_encoder_strides = [
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2]
    ]

    video_decoder_filters = list(reversed(video_encoder_filters))
    video_decoder_kernel_size = video_encoder_kernel_size
    video_decoder_strides = list(reversed(video_encoder_strides))

    video_discriminator_filters = video_encoder_filters
    video_discriminator_kernel_size = video_encoder_kernel_size
    video_discriminator_strides = video_encoder_strides
    video_discriminator_intermediate_size = 64
    # endregion

    # region Audio
    audio_encoder_filters = [32, 48, 64, 96, 128]
    audio_encoder_kernel_size = [31, 21, 13, 9, 5]
    audio_encoder_strides = [6, 5, 4, 4, 4]

    audio_decoder_filters = list(reversed(audio_encoder_filters))
    audio_decoder_kernel_size = list(reversed(audio_encoder_kernel_size))
    audio_decoder_strides = list(reversed(audio_encoder_strides))

    audio_discriminator_filters = audio_encoder_filters
    audio_discriminator_kernel_size = audio_encoder_kernel_size
    audio_discriminator_strides = audio_encoder_strides
    audio_discriminator_intermediate_size = video_discriminator_intermediate_size
    # endregion
    # endregion

    # region Model building
    code_shape = (code_length, code_size)
    reshape_to_common_code = Reshape(target_shape=code_shape, name="ReshapeToCodeShape")

    # region Video models
    # region Video encoder
    video_shape = (video_length, video_height, video_width, video_channels)
    video_encoder_layers = get_encoder_layers(rank=3,
                                              filters=video_encoder_filters,
                                              kernel_size=video_encoder_kernel_size,
                                              strides=video_encoder_strides,
                                              code_size=code_size,
                                              code_activation="tanh",
                                              mode="conv",
                                              name="VideoEncoder",
                                              flatten_code=False,
                                              use_code_bias=True,
                                              )
    video_code_shape = compute_output_shape(video_encoder_layers, video_shape)
    video_encoder_layers.append(reshape_to_common_code)
    video_encoder = to_sequential(layers=video_encoder_layers, input_shape=video_shape, name="VideoEncoder")
    # endregion

    # region Video decoder
    video_decoder_layers = get_decoder_layers(rank=3,
                                              filters=video_decoder_filters,
                                              kernel_size=video_decoder_kernel_size,
                                              strides=video_decoder_strides,
                                              channels=video_channels,
                                              output_activation="linear",
                                              mode="conv",
                                              name="VideoDecoder",
                                              stem_kernel_size=7)
    reshape_to_video_code = Reshape(target_shape=video_code_shape, name="ReshapeToVideoCodeShape")
    video_decoder_layers.insert(0, reshape_to_video_code)
    video_decoder = to_sequential(layers=video_decoder_layers, input_shape=code_shape, name="VideoDecoder")
    # endregion

    # region Video discriminator
    video_discriminator = make_discriminator(input_shape=video_shape,
                                             filters=video_discriminator_filters,
                                             kernel_size=video_discriminator_kernel_size,
                                             strides=video_discriminator_strides,
                                             intermediate_size=video_discriminator_intermediate_size,
                                             intermediate_activation="relu",
                                             include_intermediate_output=False,
                                             name="VideoDiscriminator",
                                             mode="conv")
    # endregion

    video_generator = AE(encoder=video_encoder, decoder=video_decoder, name="VideoGenerator")
    # endregion

    # region Audio models

    # region Video encoder
    audio_shape = (audio_length, audio_channels)
    audio_encoder_layers = get_encoder_layers(rank=1,
                                              filters=audio_encoder_filters,
                                              kernel_size=audio_encoder_kernel_size,
                                              strides=audio_encoder_strides,
                                              code_size=code_size,
                                              code_activation="tanh",
                                              mode="conv",
                                              name="AudioEncoder",
                                              flatten_code=False,
                                              use_code_bias=True,
                                              )
    audio_code_shape = compute_output_shape(audio_encoder_layers, audio_shape)
    audio_encoder_layers.append(reshape_to_common_code)
    audio_encoder = to_sequential(layers=audio_encoder_layers, input_shape=audio_shape, name="AudioEncoder")
    # endregion

    # region Audio decoder
    audio_decoder_layers = get_decoder_layers(rank=1,
                                              filters=audio_decoder_filters,
                                              kernel_size=audio_decoder_kernel_size,
                                              strides=audio_decoder_strides,
                                              channels=audio_channels,
                                              output_activation="linear",
                                              mode="conv",
                                              name="AudioDecoder",
                                              stem_kernel_size=7)
    reshape_to_audio_code = Reshape(target_shape=audio_code_shape, name="ReshapeToAudioCodeShape")
    audio_decoder_layers.insert(0, reshape_to_audio_code)
    audio_decoder = to_sequential(layers=audio_decoder_layers, input_shape=code_shape, name="AudioDecoder")
    # endregion

    # region Audio discriminator
    audio_discriminator = make_discriminator(input_shape=audio_shape,
                                             filters=audio_discriminator_filters,
                                             kernel_size=audio_discriminator_kernel_size,
                                             strides=audio_discriminator_strides,
                                             intermediate_size=audio_discriminator_intermediate_size,
                                             intermediate_activation="relu",
                                             include_intermediate_output=False,
                                             name="AudioDiscriminator",
                                             mode="conv")
    # endregion

    audio_generator = AE(encoder=audio_encoder, decoder=audio_decoder, name="AudioGenerator")
    # endregion

    generators_optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    discriminators_optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    model = CoupledVAEGANs(generator_1=audio_generator,
                           generator_2=video_generator,
                           discriminator_1=audio_discriminator,
                           discriminator_2=video_discriminator,
                           generators_optimizer=generators_optimizer,
                           discriminators_optimizer=discriminators_optimizer,
                           base_reconstruction_loss_weight=1e+2,
                           base_divergence_loss_weight=1e-1,
                           cycle_reconstruction_loss_weight=1e+2,
                           cycle_divergence_loss_weight=1e-1,
                           adversarial_loss_weight=1e+0,
                           gradient_penalty_loss_weight=1e+1,
                           domain_1_name="Audio",
                           domain_2_name="Video",
                           seed=seed,
                           name="EmolyCoupledVAEGANs")
    # endregion

    # region Protocol
    protocol = Protocol(model=model,
                        dataset_name="emoly",
                        protocol_name="audio_video",
                        output_range=(-1.0, 1.0),
                        seed=seed,
                        base_log_dir="../logs/tests/emoly_coupled_vaegans"
                        )

    # region Training
    video_preprocess = make_video_preprocessor(to_grayscale=video_channels == 1,
                                               activation_range="tanh",
                                               include_labels=False,
                                               target_size=(video_height, video_width))

    def preprocess(inputs):
        audio, video, faces = inputs
        _, height, width, _ = tf.unstack(tf.shape(video))

        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)

        faces.set_shape((None, 4))
        faces = tf.clip_by_value(faces, 0.0, 1.0)
        offset_height, target_height, offset_width, target_width = tf.unstack(faces, axis=-1)

        offset_height = tf.reduce_min(offset_height)
        offset_width = tf.reduce_min(offset_width)
        target_height = tf.reduce_max(target_height)
        target_width = tf.reduce_max(target_width)

        target_height = tf.cast(height * (target_height - offset_height), tf.int32)
        target_width = tf.cast(width * (target_width - offset_width), tf.int32)
        offset_height = tf.cast(height * offset_height, tf.int32)
        offset_width = tf.cast(width * offset_width, tf.int32)

        video = tf.image.crop_to_bounding_box(video, offset_height, offset_width, target_height, target_width)
        # noinspection PyArgumentList
        video = video_preprocess(video)

        return audio, video

    def augment(inputs):
        audio, video = preprocess(inputs)

        # video = tf.image.random_flip_up_down(video, seed=seed)
        video = tf.image.random_flip_left_right(video, seed=seed)

        # audio += tf.random.normal(tf.shape(audio), stddev=0.1, seed = seed)
        # audio += tf.random.normal(tf.shape(audio), stddev=0.1, seed = seed)

        return audio, video

    train_pattern = Pattern(
        (
            ModalityLoadInfo(RawAudio, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
            ModalityLoadInfo(Faces, length=video_length),
        ),
        preprocessor=augment
    )
    # endregion

    # region Callbacks
    test_pattern = Pattern(
        (
            ModalityLoadInfo(RawAudio, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
            ModalityLoadInfo(Faces, length=video_length),
        ),
        preprocessor=preprocess
    )

    # region AUC callback
    def auc_preprocess(inputs, labels):
        audio, video = preprocess(inputs)
        return (audio, video), labels

    auc_pattern = test_pattern.with_labels()
    auc_pattern.output_map = auc_preprocess

    def audio_video_reconstruction_score(y_true, y_pred):
        true_audio, true_video = y_true
        pred_audio, pred_video = y_pred

        reduction_axis_audio = tuple(range(2, true_audio.shape.rank))
        reduction_axis_video = tuple(range(2, true_video.shape.rank))

        error_audio = tf.reduce_mean(tf.abs(true_audio - pred_audio), axis=reduction_axis_audio)
        error_video = tf.reduce_mean(tf.abs(true_video - pred_video), axis=reduction_axis_video)

        error_audio = tf.expand_dims(error_audio, axis=-1)
        error_audio = tf.nn.avg_pool1d(error_audio,
                                       ksize=audio_length_multiplier,
                                       strides=audio_length_multiplier,
                                       padding="SAME")
        error_audio = tf.squeeze(error_audio)

        error = (error_audio + error_video) * 0.5
        error.set_shape((None, video_length))
        return error

    auc_callback_config = AUCCallbackConfig(base_model=model, pattern=auc_pattern, base_function=model.autoencode,
                                            labels_length=video_length, prefix="", sample_count=2048,
                                            convert_to_io_compare_model=True,
                                            io_compare_metrics=audio_video_reconstruction_score)
    auc_callback_configs = [auc_callback_config]
    # endregion
    # region Modality callbacks
    autoencode_video_callback_config = ImageCallbackConfig(autoencoder=model.autoencode,
                                                           pattern=test_pattern,
                                                           is_train_callback=True,
                                                           name="video_autoencode_train",
                                                           modality_indices=1,
                                                           )

    cross_autoencode_video_callback_config = ImageCallbackConfig(autoencoder=model.cross_autoencode,
                                                                 pattern=test_pattern,
                                                                 is_train_callback=True,
                                                                 name="video_cross_autoencode_train",
                                                                 modality_indices=1,
                                                                 )

    # autoencode_audio_callback_config = ImageCallbackConfig(autoencoder=model.autoencode,
    #                                                        pattern=test_pattern,
    #                                                        is_train_callback=True,
    #                                                        name="audio_autoencode_train",
    #                                                        modality_indices=0,
    #                                                        )
    #
    # cross_autoencode_audio_callback_config = ImageCallbackConfig(autoencoder=model.cross_autoencode,
    #                                                              pattern=test_pattern,
    #                                                              is_train_callback=True,
    #                                                              name="audio_cross_autoencode_train",
    #                                                              modality_indices=0,
    #                                                              )

    modality_callback_configs = [
        autoencode_video_callback_config,
        # autoencode_audio_callback_config,
        cross_autoencode_video_callback_config,
        # cross_autoencode_audio_callback_config,
    ]

    # endregion
    # endregion

    train_config = ProtocolTrainConfig(batch_size=batch_size,
                                       pattern=train_pattern,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       initial_epoch=initial_epoch,
                                       validation_steps=validation_steps,
                                       modality_callback_configs=modality_callback_configs,
                                       auc_callback_configs=auc_callback_configs,
                                       save_frequency="epoch")

    protocol.train_model(train_config)
    # endregion
