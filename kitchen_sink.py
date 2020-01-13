import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Reshape, Concatenate, Lambda, Add

from models.energy_based.BMEG import BMEG, ModalityModels
from CustomKerasLayers import ResBlock1D, ResBlock3D, ResBlock1DTranspose, ResBlock3DTranspose
from modalities import Pattern, ModalityLoadInfo, RawVideo, MelSpectrogram, Faces
from preprocessing.video_preprocessing import make_video_preprocess
from protocols import Protocol, ProtocolTrainConfig
from callbacks.configs import AudioVideoCallbackConfig


def get_audio_encoder(length, channels, code_size, name="AudioEncoder") -> Sequential:
    initializer = ResBlock1D.get_fixup_initializer(model_depth=15)
    shared_params = {"kernel_size": 3, "use_batch_norm": False, "activation": "selu", "kernel_initializer": initializer}
    layers = [
        ResBlock1D(filters=128, basic_block_count=1, strides=1, input_shape=(length, channels), **shared_params),
        ResBlock1D(filters=128, basic_block_count=4, strides=2, **shared_params),
        ResBlock1D(filters=64, basic_block_count=4, strides=2, **shared_params),
        ResBlock1D(filters=code_size, basic_block_count=1, strides=1, **shared_params),
    ]
    audio_encoder = Sequential(layers, name=name)
    return audio_encoder


def get_audio_generator(length, channels, code_size, name="AudioGenerator") -> Sequential:
    initializer = ResBlock1D.get_fixup_initializer(model_depth=15)
    shared_params = {"kernel_size": 3, "use_batch_norm": False, "activation": "selu", "kernel_initializer": initializer,
                     "use_residual_bias": False, "use_conv_bias": True}
    layers = [
        ResBlock1DTranspose(filters=64, basic_block_count=1, strides=1, input_shape=(length, code_size),
                            **shared_params),
        ResBlock1DTranspose(filters=128, basic_block_count=4, strides=2, **shared_params),
        ResBlock1DTranspose(filters=128, basic_block_count=4, strides=2, **shared_params),
        Dense(units=channels, activation="tanh", kernel_initializer="he_normal"),
    ]
    audio_encoder = Sequential(layers, name=name)
    return audio_encoder


def get_audio_decoder(length, channels, code_size, name="AudioDecoder") -> Sequential:
    return get_audio_generator(length, channels, code_size, name=name)


def get_video_encoder(length, height, width, channels, code_size, name="VideoEncoder") -> Sequential:
    input_shape = (length, height, width, channels)
    initializer = ResBlock1D.get_fixup_initializer(model_depth=12)
    shared_params = {"use_batch_norm": False, "activation": "selu", "kernel_initializer": initializer}

    layers = [
        ResBlock3D(filters=16, kernel_size=3, strides=(1, 2, 2), input_shape=input_shape, **shared_params),
        ResBlock3D(filters=32, kernel_size=3, strides=(1, 2, 2), **shared_params),
        # ResBlock3D(filters=32, kernel_size=3, strides=(1, 2, 2), **shared_params),
        ResBlock3D(filters=64, kernel_size=3, strides=(1, 2, 2), **shared_params),
        ResBlock3D(filters=128, kernel_size=3, strides=(1, 2, 2), **shared_params),
        ResBlock3D(filters=code_size, kernel_size=(3, 2, 2), strides=(1, 2, 2), **shared_params),
        ResBlock3D(filters=code_size, kernel_size=(3, 1, 1), strides=1, **shared_params),
        Reshape(target_shape=(length, code_size)),
    ]

    video_encoder = Sequential(layers, name=name)
    return video_encoder


def get_video_generator(length, channels, code_size, name="VideoGenerator") -> Sequential:
    input_shape = (length, code_size)
    initializer = ResBlock1D.get_fixup_initializer(model_depth=12)
    shared_params = {"use_batch_norm": False, "activation": "selu", "kernel_initializer": initializer,
                     "use_residual_bias": False, "use_conv_bias": True}

    layers = [
        Reshape(target_shape=(length, 1, 1, code_size), input_shape=input_shape),
        ResBlock3DTranspose(filters=code_size, kernel_size=(3, 2, 2), strides=(1, 2, 2), **shared_params),
        ResBlock3DTranspose(filters=128, kernel_size=3, strides=(1, 2, 2), **shared_params, ),
        ResBlock3DTranspose(filters=64, kernel_size=3, strides=(1, 2, 2), **shared_params),
        # ResBlock3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), **shared_params),
        ResBlock3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), **shared_params),
        ResBlock3DTranspose(filters=16, kernel_size=3, strides=(1, 2, 2), **shared_params),
        Dense(units=channels, activation="tanh", kernel_initializer="he_normal"),
    ]

    video_generator = Sequential(layers, name=name)
    return video_generator


def get_video_decoder(length, channels, code_size, name="VideoDecoder") -> Sequential:
    return get_video_generator(length, channels, code_size, name=name)


def get_fusion_autoencoder(length, audio_code_size, video_code_size) -> Model:
    audio_input_shape = (length, audio_code_size)
    video_input_shape = (length, video_code_size)
    initializer = ResBlock1D.get_fixup_initializer(model_depth=15)
    shared_params = {"kernel_size": 3, "use_batch_norm": False, "activation": "selu", "kernel_initializer": initializer}

    audio_input = Input(shape=audio_input_shape, name="FusionAudioInput")
    video_input = Input(shape=video_input_shape, name="FusionVideoInput")

    x = Concatenate()([audio_input, video_input])
    shortcut = x
    x = ResBlock1D(filters=128, basic_block_count=4, **shared_params)(x)
    x = ResBlock1D(filters=audio_code_size + video_code_size, basic_block_count=2, **shared_params)(x)
    x = Add(name="FusionShortcut")([x, shortcut])
    audio_output, video_output = Lambda(function=
                                        lambda z: tf.split(z, [audio_code_size, video_code_size], axis=-1)
                                        )(x)

    fusion_autoencoder = Model(inputs=[audio_input, video_input],
                               outputs=[audio_output, video_output],
                               name="FusionAutoencoder")
    return fusion_autoencoder


def main():
    batch_size = 16
    steps_per_epoch = 1000
    epochs = 100
    initial_epoch = 3
    validation_steps = 32

    # region Constants
    video_length = 64
    video_height = video_width = 32
    video_channels = 1

    # audio_length = video_step_length * 1920  # for wave
    audio_length = video_length * 4  # for mfcc
    # audio_channels = 2  # for wave
    audio_channels = 100  # for mfcc
    # endregion

    # region Model
    audio_code_size = 64
    video_code_size = 128

    # region Audio models
    audio_encoder = get_audio_encoder(audio_length, audio_channels, audio_code_size)
    audio_generator = get_audio_generator(video_length, audio_channels, audio_code_size)
    audio_decoder = get_audio_decoder(video_length, audio_channels, audio_code_size)

    audio_models = ModalityModels(encoder=audio_encoder,
                                  generator=audio_generator,
                                  decoder=audio_decoder,
                                  name="Audio")
    # endregion

    # region Video models
    video_encoder = get_video_encoder(video_length, video_height, video_width, video_channels, video_code_size)
    video_generator = get_video_generator(video_length, video_channels, video_code_size)
    video_decoder = get_video_decoder(video_length, video_channels, video_code_size)

    video_models = ModalityModels(encoder=video_encoder,
                                  generator=video_generator,
                                  decoder=video_decoder,
                                  name="Video")
    # endregion

    fusion_autoencoder = get_fusion_autoencoder(video_length, audio_code_size, video_code_size)

    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    generators_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model = BMEG(models_1=audio_models,
                 models_2=video_models,
                 fusion_autoencoder=fusion_autoencoder,
                 energy_margin=8e-3,
                 autoencoder_optimizer=autoencoder_optimizer,
                 generators_optimizer=generators_optimizer,
                 )
    # endregion

    # region Protocol
    protocol = Protocol(model=model,
                        dataset_name="emoly",
                        protocol_name="audio_video",
                        model_name="BMEG",
                        output_range=(-1.0, 1.0)
                        )

    # region Training
    video_preprocess = make_video_preprocess(height=video_height, width=video_width, base_channels=3)

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

    train_pattern = Pattern(
        (
            ModalityLoadInfo(MelSpectrogram, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
            ModalityLoadInfo(Faces, length=video_length),
        ),
        output_map=preprocess
    )
    # endregion

    # region Callbacks
    mel_spectrogram = protocol.dataset_config.modalities[MelSpectrogram]
    autoencode_callback_config = AudioVideoCallbackConfig(autoencoder=model.autoencode,
                                                          pattern=train_pattern,
                                                          is_train_callback=True,
                                                          name="av_autoencode_train",
                                                          modality_indices=(0, 1),
                                                          mel_spectrogram=mel_spectrogram,
                                                          )

    regenerate_callback_config = AudioVideoCallbackConfig(autoencoder=model.regenerate,
                                                          pattern=train_pattern,
                                                          is_train_callback=True,
                                                          name="av_regenerate_train",
                                                          modality_indices=(0, 1),
                                                          mel_spectrogram=mel_spectrogram,
                                                          )

    modality_callback_configs = [autoencode_callback_config, regenerate_callback_config]
    # endregion

    train_config = ProtocolTrainConfig(batch_size=batch_size,
                                       pattern=train_pattern,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       initial_epoch=initial_epoch,
                                       validation_steps=validation_steps,
                                       modality_callback_configs=modality_callback_configs
                                       )

    protocol.train_model(train_config)
    # endregion


if __name__ == "__main__":
    main()
