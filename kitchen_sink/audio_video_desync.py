import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Concatenate, Lambda, Flatten, Reshape, Add
from tensorflow.python.keras.layers import Dense, Conv1D, Conv3D
from tensorflow.python.keras.layers import TimeDistributed, GlobalAveragePooling2D, AveragePooling3D
from typing import Tuple

from protocols import Protocol, ProtocolTrainConfig
from callbacks.configs import ImageCallbackConfig
from modalities import Pattern, ModalityLoadInfo, RawVideo, MelSpectrogram
from custom_tf_models.energy_based import EBM, TakeStepESF, OffsetSequences  # , IdentityESF, SwitchSamplesESF
from CustomKerasLayers import SpatialTransformer, ResBlock1D, ResBlock3D, ResBlock1DTranspose, ResBlock3DTranspose
from CustomKerasLayers import ExpandDims
from data_processing.video_processing.video_preprocessing import make_video_preprocess


def time_to_batch(inputs):
    batch_size, time, *dimensions = tf.unstack(tf.shape(inputs))
    outputs = tf.reshape(inputs, [batch_size * time, *dimensions])
    outputs.set_shape([None, *inputs.shape[2:]])
    return outputs


def make_time_from_batch(time):
    def time_from_batch(inputs):
        _, *dimensions = tf.unstack(tf.shape(inputs))
        outputs = tf.reshape(inputs, [-1, time, *dimensions])
        outputs.set_shape([None, time, *inputs.shape[1:]])
        return outputs

    return time_from_batch


def reconstruction_energy_function(inputs):
    y_true, y_pred = inputs
    audio_true, video_true = y_true
    audio_pred, video_pred = y_pred
    audio_error = tf.reduce_mean(tf.square(audio_true - audio_pred), axis=(1, 2))
    video_error = tf.reduce_mean(tf.square(video_true - video_pred), axis=(1, 2, 3, 4))
    energy = audio_error + video_error
    energy = tf.reshape(energy, [-1, 1])
    margin = 1e-2
    energy = tf.clip_by_value(energy, 0.0, margin * 2)
    energy = (energy - margin) * (1.0 / margin)
    return energy


def main():
    step_count = 3  # Max is 3 for base length of 64 (192 - for now - shortest video has 251 frames)
    # region Constants
    video_step_length = 64
    video_width = video_height = 128
    video_channels = 1
    # audio_step_length = video_step_length * 1920  # for wave
    audio_step_length = video_step_length * 4  # for mfcc
    # audio_channels = 2  # for wave
    audio_channels = 100  # for mfcc

    audio_load_length = audio_step_length * step_count
    video_load_length = video_step_length * step_count
    # endregion

    # region Energy model

    # region Audio

    audio_input_shape = (audio_step_length, audio_channels)
    audio_input = Input(shape=audio_input_shape, name="AudioInputLayer")

    # region Wave
    # x = audio_input
    # x = ResBlock1D(filters=64, kernel_size=21, strides=5)(x)
    # x = MaxPooling1D(pool_size=4)(x)
    # x = ResBlock1D(filters=128, kernel_size=5, strides=4)(x)
    # x = ResBlock1D(filters=256, kernel_size=5, strides=4)(x)
    # x = ResBlock1D(filters=256, kernel_size=5, strides=3)(x)
    # x = ResBlock1D(filters=256, kernel_size=3, strides=2)(x)
    # endregion

    # region Mel Spectrogram
    x = audio_input
    x = ResBlock1D(filters=128, kernel_size=3, strides=1)(x)
    x = ResBlock1D(filters=128, basic_block_count=4, kernel_size=3, strides=2)(x)
    x = ResBlock1D(filters=64, basic_block_count=4, kernel_size=3, strides=2)(x)
    # endregion

    audio_output = x

    audio_model = Model(inputs=audio_input, outputs=audio_output,
                        name="AudioModel")

    # endregion

    # region Video

    video_shape = (video_step_length, video_height, video_width, video_channels)
    video_input = Input(shape=video_shape, name="VideoInputLayer")

    # region Video Self Attention

    x = video_input
    x = AveragePooling3D(pool_size=(1, 4, 4))(x)
    x = ResBlock3D(filters=4, kernel_size=(1, 3, 3), strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=32, kernel_size=(1, 3, 3), strides=(1, 2, 2))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    video_self_attention_output = x

    video_self_attention_model = Model(inputs=video_input, outputs=video_self_attention_output,
                                       name="VideoSelfAttentionModel")

    # endregion

    # region Video Joint Attention

    joint_audio_input = Input(batch_shape=audio_output.shape, name="JointAudioAttentionInputLayer")
    joint_video_input = Input(batch_shape=video_self_attention_output.shape, name="JointVideoAttentionInputLayer")

    joint_audio_attention = joint_audio_input
    # joint_audio_attention = Dense(units=64, activation="relu", kernel_initializer="he_normal")(joint_audio_attention)
    x = Concatenate()([joint_audio_attention, joint_video_input])

    # region (Localisation)

    localisation_model = Sequential(layers=[
        Dense(units=64, activation="relu", kernel_initializer="he_normal"),
        Dense(units=32, activation="relu", kernel_initializer="he_normal")
    ],
        name="JointAttentionLocalisationModel")

    # endregion

    images = Lambda(time_to_batch)(video_input)
    x = Lambda(time_to_batch)(x)
    x = SpatialTransformer(localisation_net=localisation_model, output_size=(32, 32))([x, images])
    time_from_batch = make_time_from_batch(video_step_length)
    x = Lambda(time_from_batch)(x)

    joint_attention_output = x
    joint_attention_model = Model(inputs=[joint_audio_input, joint_video_input, video_input],
                                  outputs=joint_attention_output,
                                  name="JointAttentionModel")

    # endregion

    # region Video (Focused)

    focused_video_input = Input(batch_shape=joint_attention_model.output_shape, name="FocusedVideoInputLayer")
    x = focused_video_input
    x = ResBlock3D(filters=8, kernel_size=3, strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=16, kernel_size=3, strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=32, kernel_size=3, strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=64, kernel_size=3, strides=(1, 2, 2))(x)
    x = ResBlock3D(filters=128, kernel_size=3, strides=(1, 2, 2))(x)
    x = Reshape(target_shape=(video_step_length, 128))(x)
    focused_video_output = x

    focused_video_model = Model(inputs=focused_video_input, outputs=focused_video_output,
                                name="FocusedVideoModel")

    # endregion

    # endregion

    # region Fusion

    audio_latent_code_input = Input(batch_shape=audio_model.output_shape, name="AudioLatentCodeInputLayer")
    video_latent_code_input = Input(batch_shape=focused_video_model.output_shape, name="VideoLatentCodeInputLayer")
    x = Concatenate()([audio_latent_code_input, video_latent_code_input])
    x = ResBlock1D(filters=128, kernel_size=3)(x)
    x = ResBlock1D(filters=256, kernel_size=3)(x)
    x = ResBlock1D(filters=128, kernel_size=3)(x)
    intermediate_fusion_output = x
    x = ResBlock1D(filters=64, kernel_size=3)(x)
    x = ResBlock1D(filters=32, kernel_size=3)(x)
    x = Dense(units=1, activation=None, kernel_initializer="he_normal")(x)
    x = Flatten()(x)
    fusion_output = x

    late_fusion_model = Model(inputs=[audio_latent_code_input, video_latent_code_input],
                              outputs=[intermediate_fusion_output, fusion_output],
                              name="LateFusionModel")

    # endregion

    # region Reconstruction

    reconstruction_input = Input(batch_shape=intermediate_fusion_output.shape, name="ReconstructionInput")

    # region Audio

    # region Wave
    # x = reconstruction_input
    # x = ResBlock1DTranspose(filters=256, kernel_size=9, strides=6)(x)
    # x = ResBlock1DTranspose(filters=128, kernel_size=5, strides=4)(x)
    # x = ResBlock1DTranspose(filters=128, kernel_size=5, strides=4)(x)
    # x = ResBlock1DTranspose(filters=64, kernel_size=5, strides=4)(x)
    # x = ResBlock1DTranspose(filters=64, kernel_size=7, strides=5)(x)
    # x = ResBlock1DTranspose(filters=audio_channels, kernel_size=1, strides=1)(x)
    # reconstructed_audio = x
    # endregion

    # region Mel Spectrogram
    x = reconstruction_input
    x = ResBlock1DTranspose(filters=128, basic_block_count=4, kernel_size=3, strides=2)(x)
    x = ResBlock1DTranspose(filters=128, basic_block_count=4, kernel_size=3, strides=2)(x)
    x = Conv1D(filters=audio_channels, kernel_size=1, strides=1, kernel_initializer="he_normal")(x)
    reconstructed_audio = x
    # endregion

    # endregion

    # region Video

    x = reconstruction_input
    x = ExpandDims(dims=[1, 2])(x)
    x = ResBlock3DTranspose(filters=128, kernel_size=(1, 5, 5), strides=(1, 4, 4))(x)
    x = ResBlock3DTranspose(filters=64, kernel_size=(1, 5, 5), strides=(1, 4, 4))(x)
    x = ResBlock3DTranspose(filters=32, kernel_size=5, strides=(1, 4, 4))(x)
    x = ResBlock3DTranspose(filters=16, kernel_size=3, strides=(1, 2, 2))(x)
    x = Conv3D(filters=video_channels, kernel_size=1, strides=1, kernel_initializer="he_normal")(x)
    reconstructed_video = x

    # endregion

    reconstruction_model = Model(inputs=reconstruction_input,
                                 outputs=[reconstructed_audio, reconstructed_video],
                                 name="ReconstructionModel")

    # endregion

    audio_latent_code = audio_model(audio_input)
    video_self_attention = video_self_attention_model(video_input)
    focused_video = joint_attention_model([audio_latent_code, video_self_attention, video_input])
    video_latent_code = focused_video_model(focused_video)
    fused_latent_code, logits = late_fusion_model([audio_latent_code, video_latent_code])
    reconstructed = reconstruction_model(fused_latent_code)
    reconstruction_energy = Lambda(function=reconstruction_energy_function)([[audio_input, video_input], reconstructed])
    logits = Add()([logits, reconstruction_energy])

    energy_model = Model(inputs=[audio_input, video_input], outputs=logits, name="EnergyModel")

    # endregion

    energy_state_functions = [
        TakeStepESF(step_count=step_count),
        OffsetSequences(step_count=step_count)
        # IdentityESF(),
        # SwitchSamplesESF(),
    ]

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.95, staircase=True)
    from misc_utils.train_utils import CyclicSchedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=CyclicSchedule(1000, 1e-5, 1e-4))

    model = EBM(energy_model=energy_model,
                energy_state_functions=energy_state_functions,
                optimizer=optimizer,
                energy_margin=1.0,
                energy_model_uses_ground_truth=False,
                name="ebm"
                )

    # region Protocol
    protocol = Protocol(model=model,
                        dataset_name="emoly",
                        protocol_name="audio_video",
                        output_range=(0.0, 1.0),
                        base_log_dir="../logs/tests/audio_video_desync"
                        )

    video_preprocess = make_video_preprocess(to_grayscale=True,
                                             activation_range="sigmoid",
                                             target_size=(video_height, video_width))

    def preprocess(inputs: Tuple[tf.Tensor, tf.Tensor]):
        audio, video = inputs
        # noinspection PyArgumentList
        video = video_preprocess(video)
        return audio, video

    train_pattern = Pattern(
        (
            ModalityLoadInfo(MelSpectrogram, length=audio_load_length),
            ModalityLoadInfo(RawVideo, length=video_load_length),
        ),
        preprocessor=preprocess
    )
    # endregion

    # region Callbacks
    focus_preview_model = Model(inputs=[audio_input, video_input], outputs=focused_video, name="FocusPreview")
    image_callbacks_configs = [
        ImageCallbackConfig(autoencoder=focus_preview_model,
                            pattern=Pattern(
                                (
                                    ModalityLoadInfo(MelSpectrogram, length=audio_step_length),
                                    ModalityLoadInfo(RawVideo, length=video_step_length),
                                ),
                                preprocessor=preprocess
                            ),
                            is_train_callback=True,
                            name="FocusPreview",
                            modality_index=1,
                            compare_to_ground_truth=False,
                            )
    ]
    # endregion

    train_config = ProtocolTrainConfig(batch_size=8,
                                       pattern=train_pattern,
                                       steps_per_epoch=1000,
                                       epochs=100,
                                       initial_epoch=0,
                                       validation_steps=32,
                                       modality_callback_configs=image_callbacks_configs,
                                       save_frequency="epoch"
                                       )

    protocol.train_model(train_config)


if __name__ == "__main__":
    main()
