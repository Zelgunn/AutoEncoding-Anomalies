import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Reshape, Concatenate, Lambda

from custom_tf_models.energy_based.BMEG import BMEG, ModalityModels
from CustomKerasLayers import ResBlock1D, ResBlock3D, ResBlock1DTranspose, ResBlock3DTranspose
from modalities import Pattern, ModalityLoadInfo, RawVideo, MelSpectrogram, Faces
from data_processing.video_preprocessing import make_video_preprocess
from protocols import Protocol, ProtocolTrainConfig
from callbacks.configs import AudioVideoCallbackConfig, AUCCallbackConfig


def get_audio_encoder(length: int,
                      channels: int,
                      code_size: int,
                      name="AudioEncoder",
                      n=96,
                      kernel_size=3
                      ) -> Sequential:
    shared_params = {"kernel_size": kernel_size}
    layers = [
        ResBlock1D(filters=n * 2, basic_block_count=1, strides=1, input_shape=(length, channels), **shared_params),
        ResBlock1D(filters=n * 2, basic_block_count=4, strides=2, **shared_params),
        ResBlock1D(filters=n * 1, basic_block_count=4, strides=2, **shared_params),
        Dense(units=code_size, activation=None, kernel_initializer="he_normal")
    ]

    audio_encoder = Sequential(layers, name=name)
    return audio_encoder


def get_audio_decoder(length: int,
                      channels: int,
                      code_size: int,
                      name="AudioDecoder",
                      n=192,
                      kernel_size=3
                      ) -> Sequential:
    shared_params = {"kernel_size": kernel_size}
    layers = [
        ResBlock1DTranspose(filters=code_size, basic_block_count=1, strides=1, input_shape=(length, code_size),
                            **shared_params),
        ResBlock1DTranspose(filters=n, basic_block_count=4, strides=2, **shared_params),
        ResBlock1DTranspose(filters=n, basic_block_count=4, strides=2, **shared_params),
        Dense(units=channels, activation=None, kernel_initializer="he_normal"),
    ]
    audio_encoder = Sequential(layers, name=name)
    return audio_encoder


def get_audio_generator(length: int,
                        channels: int,
                        code_size: int,
                        noise_size: int,
                        noise_code_size: int,
                        name="AudioGenerator",
                        n=256,
                        kernel_size=3
                        ) -> Model:
    base_decoder = get_audio_decoder(length=length,
                                     channels=channels,
                                     code_size=code_size + noise_code_size,
                                     name="{}_BaseDecoder".format(name),
                                     n=n,
                                     kernel_size=kernel_size)
    generator = get_generator(base_decoder=base_decoder,
                              length=length,
                              code_size=code_size,
                              noise_size=noise_size,
                              noise_code_size=noise_code_size,
                              name=name)
    return generator


def get_video_encoder(length: int,
                      height: int,
                      width: int,
                      channels: int,
                      code_size: int,
                      name="VideoEncoder",
                      n=32,
                      kernel_size=3,
                      ) -> Sequential:
    input_shape = (length, height, width, channels)

    layers = [
        ResBlock3D(filters=n * 1, kernel_size=kernel_size, strides=(1, 2, 2), input_shape=input_shape),
        ResBlock3D(filters=n * 2, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3D(filters=n * 4, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3D(filters=n * 8, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3D(filters=code_size, kernel_size=(kernel_size, 2, 2), strides=(1, 2, 2)),
        Reshape(target_shape=(length, code_size), name="FlattenVideoCode"),
        Dense(units=code_size, activation=None, kernel_initializer="he_normal")
    ]

    video_encoder = Sequential(layers, name=name)
    return video_encoder


def get_video_decoder(length: int,
                      channels: int,
                      code_size: int,
                      name="VideoDecoder",
                      n=24,
                      kernel_size=3
                      ) -> Sequential:
    input_shape = (length, code_size)

    layers = [
        Reshape(target_shape=(length, 1, 1, code_size), input_shape=input_shape),
        ResBlock3DTranspose(filters=n * 8, kernel_size=(kernel_size, 2, 2), strides=(1, 2, 2)),
        ResBlock3DTranspose(filters=n * 8, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3DTranspose(filters=n * 4, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3DTranspose(filters=n * 2, kernel_size=kernel_size, strides=(1, 2, 2)),
        ResBlock3DTranspose(filters=n * 1, kernel_size=kernel_size, strides=(1, 2, 2)),
        Dense(units=channels, activation=None, kernel_initializer="he_normal"),
    ]

    video_generator = Sequential(layers, name=name)
    return video_generator


def get_video_generator(length: int,
                        channels: int,
                        code_size: int,
                        noise_size: int,
                        noise_code_size: int,
                        name="VideoGenerator",
                        n=32,
                        kernel_size=3
                        ) -> Model:
    base_decoder = get_video_decoder(length=length,
                                     channels=channels,
                                     code_size=code_size + noise_code_size,
                                     name="{}_BaseDecoder".format(name),
                                     n=n,
                                     kernel_size=kernel_size)
    generator = get_generator(base_decoder=base_decoder,
                              length=length,
                              code_size=code_size,
                              noise_size=noise_size,
                              noise_code_size=noise_code_size,
                              name=name)

    return generator


def get_generator(base_decoder: Sequential,
                  length: int,
                  code_size: int,
                  noise_size: int,
                  noise_code_size: int,
                  name: str) -> Model:
    base_code_input = Input(shape=(length, code_size), name="{}_BaseCodeInput".format(name))
    noise_input = Input(shape=(noise_size,), name="{}_NoiseInput".format(name))

    intermediate_size = noise_size // length

    noise_code = noise_input
    noise_code = Dense(units=noise_size, activation="elu", kernel_initializer="he_normal")(noise_code)
    noise_code = Dense(units=noise_size, activation="elu", kernel_initializer="he_normal")(noise_code)
    noise_code = Reshape(target_shape=(length, intermediate_size))(noise_code)
    noise_code = ResBlock1D(filters=intermediate_size * 2)(noise_code)
    noise_code = ResBlock1D(filters=noise_code_size)(noise_code)
    full_code = Concatenate()([base_code_input, noise_code])
    outputs = base_decoder(full_code)

    generator = Model(inputs=[base_code_input, noise_input], outputs=outputs, name=name)
    return generator


def get_fusion_autoencoder(length: int,
                           audio_code_size: int,
                           video_code_size: int,
                           n=128) -> Model:
    audio_input_shape = (length, audio_code_size)
    video_input_shape = (length, video_code_size)
    shared_params = {"kernel_size": 3}

    audio_input = Input(shape=audio_input_shape, name="FusionAudioInput")
    video_input = Input(shape=video_input_shape, name="FusionVideoInput")

    n_5 = int(1.5 * n)

    x = Concatenate()([audio_input, video_input])
    x = ResBlock1D(filters=n * 1, basic_block_count=1, strides=2, **shared_params)(x)
    x = ResBlock1D(filters=n_5, basic_block_count=2, strides=2, **shared_params)(x)
    x = ResBlock1D(filters=n * 2, basic_block_count=2, strides=2, **shared_params)(x)
    x = ResBlock1DTranspose(filters=n * 2, basic_block_count=2, strides=2, **shared_params)(x)
    x = ResBlock1DTranspose(filters=n_5, basic_block_count=2, strides=2, **shared_params)(x)
    x = ResBlock1DTranspose(filters=n * 1, basic_block_count=1, strides=2, **shared_params)(x)
    x = ResBlock1D(filters=audio_code_size + video_code_size, basic_block_count=2, **shared_params)(x)
    audio_output, video_output = Lambda(function=lambda z: tf.split(z, [audio_code_size, video_code_size], axis=-1))(x)

    fusion_autoencoder = Model(inputs=[audio_input, video_input],
                               outputs=[audio_output, video_output],
                               name="FusionAutoencoder")
    return fusion_autoencoder


def main():
    initial_epoch = 0

    # region Constants
    video_length = 64
    video_height = video_width = 32
    video_channels = 3

    # audio_length_multiplier = 1920  # for wave
    audio_length_multiplier = 4  # for mfcc
    audio_length = video_length * audio_length_multiplier
    # audio_channels = 2  # for wave
    audio_channels = 100  # for mfcc
    # endregion

    # region Hyper-parameters
    batch_size = 12
    steps_per_epoch = 1000
    epochs = 100
    validation_steps = 64
    seed = 42

    # from misc_utils.train_utils import WarmupSchedule

    autoencoder_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5,
                                                                             decay_steps=100000,
                                                                             decay_rate=0.5,
                                                                             staircase=True)

    generators_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5,
                                                                            decay_steps=100000,
                                                                            decay_rate=0.5,
                                                                            staircase=True)
    # generators_lr_schedule = WarmupSchedule(10000, generators_lr_schedule)

    noise_size = 256
    audio_code_size = 64
    audio_noise_code_size = 32
    video_code_size = 128
    video_noise_code_size = 64
    # endregion

    # region Model

    # region Audio models
    audio_encoder = get_audio_encoder(audio_length, audio_channels, audio_code_size, n=96)
    audio_generator = get_audio_generator(video_length, audio_channels, video_code_size, noise_size,
                                          audio_noise_code_size, n=256)
    audio_decoder = get_audio_decoder(video_length, audio_channels, audio_code_size, n=196)

    audio_models = ModalityModels(encoder=audio_encoder,
                                  generator=audio_generator,
                                  decoder=audio_decoder,
                                  name="Audio")
    # endregion

    # region Video models
    video_encoder = get_video_encoder(video_length, video_height, video_width, video_channels, video_code_size, n=32)
    video_generator = get_video_generator(video_length, video_channels, audio_code_size, noise_size,
                                          video_noise_code_size, n=32)
    video_decoder = get_video_decoder(video_length, video_channels, video_code_size, n=24)

    video_models = ModalityModels(encoder=video_encoder,
                                  generator=video_generator,
                                  decoder=video_decoder,
                                  name="Video")
    # endregion

    fusion_autoencoder = get_fusion_autoencoder(video_length, audio_code_size, video_code_size)

    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=autoencoder_lr_schedule)
    generators_optimizer = tf.keras.optimizers.Adam(learning_rate=generators_lr_schedule)

    model = BMEG(models_1=audio_models,
                 models_2=video_models,
                 fusion_autoencoder=fusion_autoencoder,
                 energy_margin=1e-2,
                 autoencoder_optimizer=autoencoder_optimizer,
                 generators_optimizer=generators_optimizer,
                 )
    # endregion

    # region Protocol
    protocol = Protocol(model=model,
                        dataset_name="emoly",
                        protocol_name="audio_video",
                        model_name="BMEG",
                        output_range=(-1.0, 1.0),
                        seed=seed,
                        )

    # region Training
    video_preprocess = make_video_preprocess(height=video_height, width=video_width, to_grayscale=video_channels == 1)

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
            ModalityLoadInfo(MelSpectrogram, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
            ModalityLoadInfo(Faces, length=video_length),
        ),
        preprocessor=augment
    )
    # endregion

    # region Callbacks
    test_pattern = Pattern(
        (
            ModalityLoadInfo(MelSpectrogram, length=audio_length),
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

    auc_callback_config = AUCCallbackConfig(base_model=model.autoencode, pattern=auc_pattern,
                                            labels_length=video_length, prefix="", sample_count=2048,
                                            convert_to_io_compare_model=True,
                                            io_compare_metrics=audio_video_reconstruction_score)

    auc_callback_configs = [auc_callback_config]
    # endregion
    # region Modality callbacks
    mel_spectrogram = protocol.dataset_config.modalities[MelSpectrogram]
    autoencode_callback_config = AudioVideoCallbackConfig(autoencoder=model.autoencode,
                                                          pattern=test_pattern,
                                                          is_train_callback=True,
                                                          name="av_autoencode_train",
                                                          modality_indices=(0, 1),
                                                          mel_spectrogram=mel_spectrogram,
                                                          )

    regenerate_callback_config = AudioVideoCallbackConfig(autoencoder=model.regenerate,
                                                          pattern=test_pattern,
                                                          is_train_callback=True,
                                                          name="av_regenerate_train",
                                                          modality_indices=(0, 1),
                                                          mel_spectrogram=mel_spectrogram,
                                                          )

    modality_callback_configs = [autoencode_callback_config, regenerate_callback_config]

    # endregion
    # endregion

    train_config = ProtocolTrainConfig(batch_size=batch_size,
                                       pattern=train_pattern,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       initial_epoch=initial_epoch,
                                       validation_steps=validation_steps,
                                       modality_callback_configs=modality_callback_configs,
                                       auc_callback_configs=auc_callback_configs
                                       )

    protocol.train_model(train_config)
    # endregion


if __name__ == "__main__":
    main()
