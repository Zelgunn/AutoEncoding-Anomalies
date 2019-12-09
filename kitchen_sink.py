import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, Conv2D, AveragePooling2D, Flatten, Dense, AveragePooling1D
from typing import List, Tuple, Callable

from models.unet import UNet, AudioVideoUNet
from models.energy_based import TakeStepESF, OffsetSequences, SwitchSamplesESF, CombineESF, EBAE
from protocols import Protocol, ProtocolTrainConfig, ImageCallbackConfig, AudioCallbackConfig  # , AUCCallbackConfig
from modalities import Pattern, ModalityLoadInfo, MelSpectrogram, RawVideo
from preprocessing.video_preprocessing import make_video_preprocess
from misc_utils.train_utils import WarmupSchedule

output_min = 0.0
output_max = 1.0
gaussian_noise_factor = 4e-2


def main():
    base_step_size = 64

    video_step_size = base_step_size
    video_height = 128
    video_width = 128
    video_channels = 1
    audio_step_size = base_step_size * 4
    audio_channels = 100
    step_count = 3

    autoencoder = make_unet(base_step_size, video_height, video_width, video_channels, audio_channels)

    energy_state_functions = [
        TakeStepESF(step_count),
        OffsetSequences(step_count),
        CombineESF([TakeStepESF(step_count), SwitchSamplesESF()])
    ]

    ebae = EBAE(autoencoder=autoencoder,
                energy_margin=1e-3,
                energy_state_functions=energy_state_functions,
                name="EBAE"
                )
    # endregion

    protocol = Protocol(model=ebae, dataset_name="emoly", protocol_name="audio_video", model_name="ebae",
                        output_range=(output_min, output_max))

    # region Train pattern
    video_augment = make_video_preprocess(height=video_height,
                                          width=video_width,
                                          channels=3)

    train_pattern = Pattern(
        ModalityLoadInfo(MelSpectrogram, audio_step_size * step_count),
        ModalityLoadInfo(RawVideo, video_step_size * step_count),
        output_map=lambda audio, video: (audio, video_augment(video))
    )
    # endregion

    # region Image callback configs
    video_preprocess = make_video_preprocess(height=video_height,
                                             width=video_width,
                                             channels=3)

    image_callback_pattern = Pattern(
        ModalityLoadInfo(MelSpectrogram, audio_step_size),
        ModalityLoadInfo(RawVideo, video_step_size),
        output_map=lambda audio, video: (audio, video_preprocess(video))
    )

    train_image_callback_config = ImageCallbackConfig(autoencoder=autoencoder,
                                                      pattern=image_callback_pattern,
                                                      is_train_callback=True,
                                                      name="train",
                                                      modality_index=1)

    test_image_callback_config = ImageCallbackConfig(autoencoder=autoencoder,
                                                     pattern=image_callback_pattern,
                                                     is_train_callback=False,
                                                     name="test",
                                                     modality_index=1)

    image_callback_configs = [train_image_callback_config, test_image_callback_config]

    # endregion

    # region Audio callback config

    audio_callback_config = AudioCallbackConfig(autoencoder=autoencoder,
                                                pattern=image_callback_pattern,
                                                is_train_callback=True,
                                                name="train",
                                                modality_index=0,
                                                mel_spectrogram=protocol.dataset_config.modalities[MelSpectrogram]
                                                )

    audio_callback_configs = [audio_callback_config]

    # endregion

    # region AUC callback config
    # def preprocess_audio_video_labels(audio, video, labels):
    #     inputs = (audio, video_preprocess(video))
    #     return inputs, labels

    # miae_auc_callback_config = AUCCallbackConfig(autoencoder=autoencoder,
    #                                              pattern=Pattern(
    #                                                  ModalityLoadInfo(MelSpectrogram, audio_step_size),
    #                                                  ModalityLoadInfo(RawVideo, video_step_size),
    #                                                  output_map=preprocess_audio_video_labels
    #                                              ).with_labels(),
    #                                              labels_length=1,
    #                                              prefix="AudioVideo"
    #                                              )
    #
    # auc_callbacks_configs = [miae_auc_callback_config]
    auc_callbacks_configs = []
    # endregion

    train_config = ProtocolTrainConfig(batch_size=16,
                                       pattern=train_pattern,
                                       steps_per_epoch=1000,
                                       epochs=1000,
                                       initial_epoch=0,
                                       validation_steps=128,
                                       image_callbacks_configs=image_callback_configs,
                                       audio_callbacks_configs=audio_callback_configs,
                                       auc_callbacks_configs=auc_callbacks_configs
                                       )

    protocol.train_model(train_config)


# region High energy functions
def make_offset_mods(step_count: int):
    # Only for 2 modalities at the moment
    @tf.function
    def offset_mods(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        direction = coin_flip()
        mod_1, mod_2 = inputs
        if direction:
            mod_1, mod_2 = offset_mod_2(mod_1, mod_2)
        else:
            mod_2, mod_1 = offset_mod_2(mod_2, mod_1)
        return [mod_1, mod_2]

    def offset_mod_2(mod_1: tf.Tensor, mod_2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mod_1_length = tf.shape(mod_1)[1]
        mod_2_length = tf.shape(mod_2)[1]

        mod_1_step_size = mod_1_length // step_count
        mod_2_step_size = mod_2_length // step_count

        mod_1 = mod_1[:, :mod_1_step_size]

        min_offset = mod_2_step_size // 2
        max_offset = mod_2_length - mod_2_step_size
        random_offset = tf.random.uniform(shape=[], minval=min_offset, maxval=max_offset + 1, dtype=tf.int32)
        mod_2 = mod_2[:, random_offset:random_offset + mod_2_step_size]

        return mod_1, mod_2

    return offset_mods


@tf.function
def flip_mod(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    mod_1, mod_2 = inputs
    flipped_mod = coin_flip()
    if flipped_mod:
        mod_1 = tf.reverse(mod_1, axis=(1,))
    else:
        mod_2 = tf.reverse(mod_2, axis=(1,))
    return [mod_1, mod_2]


@tf.function
def split_mod(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    mod_1, mod_2 = inputs

    split_mod_1 = coin_flip()
    if split_mod_1:
        mod_1 = split_one_mod(mod_1)
    else:
        mod_2 = split_one_mod(mod_2)
    return [mod_1, mod_2]


def split_one_mod(mod: tf.Tensor) -> tf.Tensor:
    length = tf.shape(mod)[1]
    half_length = length // 2

    split_start = tf.random.uniform(shape=[], minval=1, maxval=half_length - 1, dtype=tf.int32)
    split_end = split_start + half_length

    min_weight = 1.0 / (tf.cast(half_length, tf.float32) - 1.0)
    weights = tf.linspace(min_weight, 1.0 - min_weight, half_length)
    weights_shape = [1, half_length] + [1] * (len(mod.shape) - 2)
    weights = tf.reshape(weights, weights_shape)

    start = mod[:, split_start - 1]
    end = mod[:, split_end + 1]
    start = tf.expand_dims(start, axis=1)
    end = tf.expand_dims(end, axis=1)
    delta = end - start
    interpolated = start + delta * weights

    noise = tf.random.normal(shape=weights_shape) * gaussian_noise_factor
    interpolated = tf.clip_by_value(interpolated + noise, output_min, output_max)

    result = tf.concat([mod[:, :split_start], interpolated, mod[:, split_end:]], axis=1)
    return result


@tf.function
def switch_mod_samples(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    mod_1, mod_2 = inputs
    mod_2 = tf.reverse(mod_2, axis=(0,))
    return [mod_1, mod_2]


def make_resample_mods(step_count: int):
    max_resampling_factor = log2_int(step_count)

    @tf.function
    def resample_mods(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        index = tf.random.uniform(shape=[], minval=0, maxval=len(inputs), dtype=tf.int32)
        outputs = []
        for i in range(len(inputs)):
            x = inputs[i]
            shape = x.shape
            new_length = shape[1] // step_count if shape[1] is not None else None
            shape = (shape[0], new_length, *shape[2:])
            if i == index:
                output = resample_mod(x)
            else:
                output = take_mod_step(x, step_count)
            output.set_shape(shape)
            outputs.append(output)
        return outputs

    def resample_mod(mod: tf.Tensor) -> tf.Tensor:
        resampling_factor = tf.random.uniform(shape=[], minval=1, maxval=max_resampling_factor, dtype=tf.int32)
        resampling_factor = tf.pow(2, resampling_factor)
        rank = len(mod.shape)
        begin = [0] * rank
        end = tf.shape(mod)
        strides = [1, resampling_factor] + [1] * (rank - 2)
        mod = tf.strided_slice(mod, begin=begin, end=end, strides=strides)

        remaining_steps = step_count // resampling_factor
        if remaining_steps > 1:
            mod = take_mod_step(mod, remaining_steps)

        return mod

    return resample_mods


# endregion

# region Low Energy Functions

@tf.function
def add_gaussian_noise(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    outputs = []
    for x in inputs:
        if coin_flip():
            noise: tf.Tensor = tf.random.normal(tf.shape(x)) * gaussian_noise_factor
            x = tf.clip_by_value(x + noise, output_min, output_max)
        outputs.append(x)
    return outputs


@tf.function
def remove_modality(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    index = tf.random.uniform(shape=[], minval=0, maxval=len(inputs), dtype=tf.int32)
    outputs = []
    for i in range(len(inputs)):
        if i == index:
            output = tf.zeros_like(inputs[i])
        else:
            output = inputs[i]
        outputs.append(output)
    return outputs


# endregion

# region Helpers
@tf.function
def coin_flip():
    return tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.5


def make_take_step(step_count: int):
    @tf.function
    def take_step(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        outputs = []
        for x in inputs:
            x = take_mod_step(x, step_count)
            outputs.append(x)
        return outputs

    return take_step


def take_mod_step(mod: tf.Tensor, step_count: int):
    step_size = tf.shape(mod)[1] // step_count
    return mod[:, :step_size]


def make_step_function(step_count: int,
                       base_function: Callable[[List[tf.Tensor]], List[tf.Tensor]]
                       ) -> Callable[[List[tf.Tensor]], List[tf.Tensor]]:
    take_step = make_take_step(step_count)

    def step_function(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        inputs = take_step(inputs)
        return base_function(inputs)

    step_function.__name__ = base_function.__name__
    return step_function


@tf.function
def log2(x):
    return tf.math.log(x) / tf.math.log(2.0)


@tf.function
def log2_int(x):
    return tf.cast(log2(tf.cast(x, tf.float32)), tf.int32)


# endregion


def make_unet(base_step_size, video_height, video_width, video_channels, audio_channels):
    video_step_size = base_step_size
    audio_step_size = base_step_size * 4

    image_input_shape = (video_height, video_width, video_channels)

    image_encoder_layers = [
        Conv2D(filters=32, kernel_size=7, strides=2, padding="same", activation="linear",
               kernel_initializer="glorot_uniform", input_shape=image_input_shape),
        AveragePooling2D(),
        Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Flatten(),
        Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    ]

    image_unet = UNet(encoder_layers=image_encoder_layers, name="ImageUNet")

    audio_input_shape = (audio_step_size, audio_channels)

    audio_encoder_layers = [
        Conv1D(filters=256, kernel_size=7, strides=2, padding="same", activation="linear",
               kernel_initializer="glorot_uniform", input_shape=audio_input_shape),
        Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        Conv1D(filters=128, kernel_size=3, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
    ]

    audio_unet = UNet(encoder_layers=audio_encoder_layers, name="AudioUNet")

    fused_latent_code_shape = (video_step_size, 256)
    time_encoder_layers = [
        Conv1D(filters=256, kernel_size=7, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal", input_shape=fused_latent_code_shape),
        AveragePooling1D(),
        Conv1D(filters=256, kernel_size=7, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
        AveragePooling1D(),
        Conv1D(filters=128, kernel_size=7, strides=2, padding="same", activation="relu",
               kernel_initializer="he_normal"),
    ]

    time_unet = UNet(encoder_layers=time_encoder_layers, name="TimeUNet")

    audio_video_unet = AudioVideoUNet(image_unet, audio_unet, time_unet, learning_rate=WarmupSchedule(1000, 4e-4))
    return audio_video_unet


if __name__ == "__main__":
    main()
