import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Reshape, Concatenate, Lambda, AveragePooling1D, UpSampling1D

from custom_tf_models import AE
from custom_tf_models.auxiliary.EBL3 import EBL3
from protocols.utils import get_encoder_layers, get_decoder_layers, to_sequential
from modalities import Pattern, ModalityLoadInfo, RawVideo, MelSpectrogram
from data_processing.video_preprocessing import make_video_preprocess
from protocols import Protocol, ProtocolTrainConfig
from callbacks.configs import ImageCallbackConfig
from CustomKerasLayers import ResBlock1D


# region Audio AE
def get_audio_encoder(length: int,
                      channels: int,
                      n: int,
                      ) -> Sequential:
    layers = get_encoder_layers(rank=2,
                                filters=[
                                    n * 1,
                                    n * 2,
                                    n * 4,
                                    n * 6,
                                    n * 8,
                                    n * 8,
                                    n * 8,
                                ],
                                kernel_size=3,
                                strides=[
                                    [2, 2],  # 128, 64
                                    [2, 2],  # 064, 32
                                    [2, 2],  # 032, 16
                                    [2, 2],  # 016, 08
                                    [2, 2],  # 008, 04
                                    [1, 2],  # 008, 02
                                    [1, 2],  # 008, 01
                                ],
                                code_size=n * 8,
                                code_activation="linear",
                                basic_block_count=1,
                                mode="conv",
                                name="AudioEncoder"
                                )

    input_shape = (length, channels, 1)
    output_shape = (length // 32, n * 8)

    layers = [
        Reshape(target_shape=input_shape),
        *layers,
        Reshape(target_shape=output_shape),
    ]

    return to_sequential(layers=layers, input_shape=(length, channels), name="AudioEncoder")


def get_audio_decoder(length: int,
                      channels: int,
                      n: int,
                      ) -> Sequential:
    layers = get_decoder_layers(rank=2,
                                filters=[
                                    n * 8,
                                    n * 8,
                                    n * 8,
                                    n * 6,
                                    n * 4,
                                    n * 2,
                                    n * 1,
                                ],
                                kernel_size=3,
                                strides=[
                                    [1, 2],  # 008, 01
                                    [1, 2],  # 008, 02
                                    [2, 2],  # 008, 04
                                    [2, 2],  # 016, 08
                                    [2, 2],  # 032, 16
                                    [2, 2],  # 064, 32
                                    [2, 2],  # 128, 64
                                ],
                                channels=1,
                                output_activation="linear",
                                basic_block_count=1,
                                mode="conv",
                                name="AudioDecoder",
                                stem_kernel_size=7,
                                )

    input_shape = (length // 32, 1, n * 8)
    output_shape = (length, channels)

    layers = [
        Reshape(target_shape=input_shape),
        *layers,
        Reshape(target_shape=output_shape),
    ]

    return to_sequential(layers=layers, input_shape=(length // 32, n * 8), name="AudioDecoder")


# endregion

# region Video AE
def get_video_encoder(length: int,
                      height: int,
                      width: int,
                      channels: int,
                      n: int,
                      ) -> Sequential:
    layers = get_encoder_layers(rank=3,
                                filters=[n, int(n * 1.5), n * 2, n * 4, n * 6, n * 8],
                                kernel_size=3,
                                strides=[
                                    [2, 2, 2],  # 32, 32, 32
                                    [2, 2, 2],  # 16, 16, 16
                                    [2, 2, 2],  # 08,  8,  8
                                    [1, 2, 2],  # 08,  4,  4
                                    [1, 2, 2],  # 08,  2,  2
                                    [1, 2, 2],  # 08,  1,  1
                                ],
                                code_size=n * 8,
                                code_activation="linear",
                                basic_block_count=1,
                                mode="conv",
                                name="VideoEncoder"
                                )

    layers = [
        *layers,
        Reshape(target_shape=(length // 8, n * 8,)),
    ]

    return to_sequential(layers=layers, input_shape=(length, width, height, channels), name="VideoEncoder")


def get_video_decoder(length: int,
                      channels: int,
                      n: int,
                      ) -> Sequential:
    layers = get_decoder_layers(rank=3,
                                filters=[n * 8, n * 6, n * 4, n * 2, int(1.5 * n), n],
                                kernel_size=3,
                                strides=[
                                    [1, 2, 2],  # 08,  2,  2
                                    [1, 2, 2],  # 08,  4,  4
                                    [1, 2, 2],  # 08,  8,  8
                                    [2, 2, 2],  # 16, 16, 16
                                    [2, 2, 2],  # 32, 32, 32
                                    [2, 2, 2],  # 64, 64, 64
                                ],
                                channels=channels,
                                output_activation="linear",
                                basic_block_count=1,
                                mode="conv",
                                name="VideoDecoder",
                                stem_kernel_size=7,
                                )

    layers = [
        Reshape(target_shape=(length // 8, 1, 1, n * 8)),
        *layers,
    ]

    return to_sequential(layers=layers, input_shape=(length // 8, n * 8,), name="VideoDecoder")


# endregion

def get_fusion_autoencoder(length: int, audio_n: int, video_n: int) -> Model:
    audio_input_layer = Input(shape=(length, audio_n * 8,), name="AudioCodeInput")
    video_input_layer = Input(shape=(length, video_n * 8,), name="VideoCodeInput")

    n = audio_n + video_n

    fused_codes = Concatenate(axis=-1, name="ConcatCodes")([audio_input_layer, video_input_layer])
    fused_codes = ResBlock1D(filters=n * 8, basic_block_count=1, kernel_size=3)(fused_codes)
    fused_codes = AveragePooling1D()(fused_codes)
    fused_codes = ResBlock1D(filters=n * 8, basic_block_count=1, kernel_size=3)(fused_codes)
    fused_codes = AveragePooling1D()(fused_codes)
    fused_codes = ResBlock1D(filters=n * 8, basic_block_count=1, kernel_size=3)(fused_codes)
    fused_codes = UpSampling1D()(fused_codes)
    fused_codes = ResBlock1D(filters=n * 8, basic_block_count=1, kernel_size=3)(fused_codes)
    fused_codes = UpSampling1D()(fused_codes)

    def output_function(inputs):
        _fused_codes, base_audio_code, base_video_code = inputs
        audio_code, video_code = tf.split(_fused_codes, num_or_size_splits=[audio_n * 8, video_n * 8], axis=-1)
        return audio_code + base_audio_code, video_code + base_video_code

    outputs = Lambda(function=output_function)([fused_codes, audio_input_layer, video_input_layer])

    model = Model(inputs=[audio_input_layer, video_input_layer], outputs=outputs, name="FusionAutoencoder")
    return model


def main():
    initial_epoch = 6

    # region Constants
    video_length = 64
    video_height = video_width = 64
    video_channels = 3

    # audio_length_multiplier = 1920  # for wave
    audio_length_multiplier = 4  # for mfcc
    audio_length = video_length * audio_length_multiplier
    # audio_channels = 2  # for wave
    audio_channels = 128  # for mfcc
    # endregion

    # region Hyper-parameters
    batch_size = 4
    steps_per_epoch = 2500
    epochs = 100
    validation_steps = 128
    seed = 42

    video_n = 32
    audio_n = 16
    # endregion

    # region Model

    # region Audio models
    audio_encoder = get_audio_encoder(audio_length, audio_channels, n=audio_n)
    audio_decoder = get_audio_decoder(audio_length, audio_channels, n=audio_n)

    audio_autoencoder = AE(encoder=audio_encoder, decoder=audio_decoder, name="AudioAutoencoder")
    # endregion

    # region Video models
    video_encoder = get_video_encoder(video_length, video_height, video_width, video_channels, n=video_n)
    video_decoder = get_video_decoder(video_length, video_channels, n=video_n)

    video_autoencoder = AE(encoder=video_encoder, decoder=video_decoder, name="VideoAutoencoder")
    # endregion

    fusion_autoencoder = get_fusion_autoencoder(audio_n=audio_n, video_n=video_n, length=video_length // 8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model = EBL3(audio_autoencoder=audio_autoencoder,
                 video_autoencoder=video_autoencoder,
                 fusion_autoencoder=fusion_autoencoder,
                 energy_margin=2e-2,
                 optimizer=optimizer,
                 )
    # endregion

    # region Protocol
    protocol = Protocol(model=model,
                        dataset_name="audioset",
                        protocol_name="audio_video",
                        output_range=(-1.0, 1.0),
                        seed=seed,
                        base_log_dir="../logs/tests/audioset_ebl3"
                        )

    # region Training
    video_preprocess = make_video_preprocess(to_grayscale=video_channels == 1,
                                             activation_range="tanh",
                                             target_size=(video_height, video_width))

    def preprocess(inputs):
        audio, video = inputs
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
            ModalityLoadInfo(MelSpectrogram, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
        ),
        preprocessor=augment
    )
    # endregion

    # region Callbacks
    test_pattern = Pattern(
        (
            ModalityLoadInfo(MelSpectrogram, length=audio_length),
            ModalityLoadInfo(RawVideo, length=video_length),
        ),
        preprocessor=preprocess
    )

    # region Modality callbacks
    video_autoencode_callback_config = ImageCallbackConfig(autoencoder=model.autoencode,
                                                           pattern=test_pattern,
                                                           is_train_callback=True,
                                                           name="video_autoencode_train",
                                                           modality_indices=1,
                                                           )

    audio_autoencode_callback_config = ImageCallbackConfig(autoencoder=model.autoencode,
                                                           pattern=test_pattern,
                                                           is_train_callback=True,
                                                           name="audio_autoencode_train",
                                                           modality_indices=0,
                                                           )

    modality_callback_configs = [video_autoencode_callback_config, audio_autoencode_callback_config]

    # endregion
    # endregion

    train_config = ProtocolTrainConfig(batch_size=batch_size,
                                       pattern=train_pattern,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       initial_epoch=initial_epoch,
                                       validation_steps=validation_steps,
                                       modality_callback_configs=modality_callback_configs,
                                       auc_callback_configs=None,
                                       save_frequency="epoch"
                                       )

    protocol.train_model(train_config)
    # endregion


if __name__ == "__main__":
    main()
