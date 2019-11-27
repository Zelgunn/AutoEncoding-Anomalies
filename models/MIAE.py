import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from typing import List, Dict, Tuple
from enum import IntEnum

from models import CustomModel, IAE
from models.utils import split_steps


class MIAE(CustomModel):
    def __init__(self,
                 iaes: List[IAE],
                 learning_rate=1e-3,
                 **kwargs):
        super(MIAE, self).__init__(**kwargs)

        self.iaes = iaes
        latent_code_sizes = [np.prod(iae.encoder.output_shape[1:]) for iae in iaes]
        fusion_model = FusionModel(latent_code_sizes, mode=FusionModelMode.ONE_TO_ONE)

        fusion_base_input_layers = [Input(shape=[code_size], name="FusionInputBase_{}".format(i))
                                    for i, code_size in enumerate(latent_code_sizes)]

        fuse_with_input_layers = [Input(shape=[code_size], name="FusionInputFuseWith_{}".format(i))
                                  for i, code_size in enumerate(latent_code_sizes)]

        fusion_input_layers = [fusion_base_input_layers, fuse_with_input_layers]

        fusion_output_layers = fusion_model(fusion_input_layers)

        self.fusion_model = Model(inputs=fusion_input_layers, outputs=fusion_output_layers, name="FusionModelProxy")

        self.optimizer = None
        self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=None, mask=None):
        inputs, inputs_shapes, _ = self.split_inputs(inputs, merge_batch_and_steps=True)

        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].encode(inputs[i])
            latent_codes.append(latent_code)

        refined_latent_codes = self.fusion_model([latent_codes, latent_codes])

        outputs = []
        for i in range(self.modality_count):
            output = self.iaes[i].decode(refined_latent_codes[i])
            output = tf.reshape(output, inputs_shapes[i])
            outputs.append(output)

        return outputs

    @property
    def metrics_names(self):
        return ["reconstruction"] + [iae.name for iae in self.iaes]

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs)
            total_loss = tf.reduce_sum(losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return (total_loss, *losses)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor]:
        # outputs = self(inputs)
        #
        # losses = []
        # for i in range(self.modality_count):
        #     modality_loss: tf.Tensor = tf.reduce_mean(tf.square(inputs[i] - outputs[i]))
        #     losses.append(modality_loss)

        split_inputs, _, unmerged_shapes = self.split_inputs(inputs, merge_batch_and_steps=False)

        step_count = unmerged_shapes[0][1]
        offset = tf.random.uniform(shape=[], minval=0, maxval=step_count, dtype=tf.int32)

        original_latent_codes = []
        interpolated_latent_codes = []
        for i in range(self.modality_count):
            iae = self.iaes[i]
            modality = split_inputs[i]

            if offset == 0:
                latent_code = iae.encode(modality[:, 0])
                original_latent_code = latent_code
                interpolated_latent_code = latent_code

            elif offset == (step_count - 1):
                latent_code = iae.encode(modality[:, -1])
                original_latent_code = latent_code
                interpolated_latent_code = latent_code

            else:
                original_latent_code = iae.encode(modality[:, offset])
                factor = tf.cast(offset / (step_count - 1), tf.float32)
                start_encoded = iae.encode(modality[:, 0])
                end_encoded = iae.encode(modality[:, -1])
                interpolated_latent_code = factor * end_encoded + (1.0 - factor) * start_encoded

            original_latent_codes.append(original_latent_code)
            interpolated_latent_codes.append(interpolated_latent_code)

        refined_latent_codes = self.fusion_model([interpolated_latent_codes, original_latent_codes])

        losses = []
        for i in range(self.modality_count):
            output = self.iaes[i].decode(refined_latent_codes[i])
            target = split_inputs[i][:, offset]
            output = tf.reshape(output, tf.shape(target))
            modality_loss: tf.Tensor = tf.reduce_mean(tf.square(target - output))
            losses.append(modality_loss)

        return tuple(losses)

    @tf.function
    def interpolate(self, inputs):
        split_inputs, inputs_shape, _ = self.split_inputs(inputs, merge_batch_and_steps=True)
        original_latent_codes = self.encode_step(split_inputs)
        interpolated_latent_codes = self.get_interpolated_latent_code(inputs, merge_batch_and_steps=True)
        refined_latent_codes = self.fusion_model([interpolated_latent_codes, original_latent_codes])
        outputs = self.decode_step(refined_latent_codes)
        outputs = [tf.reshape(decoded, input_shape) for decoded, input_shape in zip(outputs, inputs_shape)]
        return outputs

    @tf.function
    def encode_step(self, inputs):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].encode(inputs[i])
            latent_codes.append(latent_code)
        return latent_codes

    @tf.function
    def decode_step(self, inputs):
        outputs = []
        for i in range(self.modality_count):
            decoded = self.iaes[i].decode(inputs[i])
            outputs.append(decoded)
        return outputs

    def get_interpolated_latent_code(self, inputs, merge_batch_and_steps):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].get_interpolated_latent_code(inputs[i], merge_batch_and_steps)
            latent_codes.append(latent_code)
        return latent_codes

    @tf.function
    def modalities_mse(self, inputs, ground_truths):
        errors = []
        for i in range(self.modality_count):
            error = tf.square(inputs[i] - ground_truths[i])
            errors.append(error)

        errors, _, _ = self.split_inputs(errors, merge_batch_and_steps=False)

        total_error = []
        factors = [1.0, 8.0]
        for i in range(self.modality_count):
            error = errors[i]
            reduction_axis = list(range(2, error.shape.rank))
            error = tf.reduce_mean(error, axis=reduction_axis) * factors[i]
            total_error.append(error)

        total_error = tf.reduce_sum(total_error, axis=0)
        return total_error

    def split_inputs(self, inputs, merge_batch_and_steps):
        split_inputs = []
        inputs_shapes = []
        new_shapes = []

        for i in range(self.modality_count):
            split_input, inputs_shape, new_shape = split_steps(inputs[i], self.step_sizes[i], merge_batch_and_steps)
            split_inputs.append(split_input)
            inputs_shapes.append(inputs_shape)
            new_shapes.append(new_shape)

        return split_inputs, inputs_shapes, new_shapes

    @property
    def modality_count(self):
        return len(self.iaes)

    @property
    def step_sizes(self) -> List[int]:
        return [iae.step_size for iae in self.iaes]

    def get_config(self):
        config = {iae.name: iae.get_config() for iae in self.iaes}
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        ids = {iae: iae.name for iae in self.iaes}
        ids[self.fusion_model] = self.fusion_model.name
        return ids

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for iae in self.iaes:
            iae.set_optimizer(optimizer)


class FusionModelMode(IntEnum):
    ALL_TO_ONE = 0,
    ONE_TO_ONE = 1


class FusionModel(Model):
    def __init__(self, latent_code_sizes: List[int], mode: FusionModelMode):
        super(FusionModel, self).__init__()

        self.latent_code_sizes = latent_code_sizes
        self.mode = mode
        self.projection_layers: List[Dense] = []

        if self.mode == FusionModelMode.ALL_TO_ONE:
            self.init_all_to_one()
        elif self.mode == FusionModelMode.ONE_TO_ONE:
            self.init_one_to_one()
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

    def init_all_to_one(self):
        for i, latent_code_size in enumerate(self.latent_code_sizes):
            layer_name = "Project_All_To_{}".format(i)
            self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def init_one_to_one(self):
        for output_mod_index, latent_code_size in enumerate(self.latent_code_sizes):
            for input_mod_index in range(len(self.latent_code_sizes)):
                if output_mod_index == input_mod_index:
                    continue
                layer_name = "Project_{}_To_{}".format(input_mod_index, output_mod_index)
                self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def call(self, inputs, training=None, mask=None):
        inputs, fuse_with = inputs

        if self.mode == FusionModelMode.ALL_TO_ONE:
            return self.call_all_to_one(inputs, fuse_with)
        elif self.mode == FusionModelMode.ONE_TO_ONE:
            return self.call_one_to_one(inputs, fuse_with)
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

    def call_all_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        fused_latent_codes = []
        for i in range(self.modality_count):
            latent_codes_to_fuse = []
            for j in range(self.modality_count):
                if i == j:
                    code = inputs_latent_codes[i]
                else:
                    code = fuse_with_latent_codes[i]
                latent_codes_to_fuse.append(code)
            fuse_with_latent_codes.append(tf.concat(latent_codes_to_fuse, axis=-1))

        outputs = []
        for i in range(self.modality_count):
            refined_latent_code = self.projection_layers[i](fused_latent_codes[i])
            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[i])
            outputs.append(refined_latent_code)

        return outputs

    def call_one_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        outputs = []
        i = 0
        for output_mod_index in range(self.modality_count):
            output_mod_latent_code = fuse_with_latent_codes[output_mod_index]
            refined_latent_code = inputs_latent_codes[output_mod_index]
            for input_mod_index in range(self.modality_count):
                if output_mod_index == input_mod_index:
                    continue
                refined_latent_code += self.projection_layers[i](output_mod_latent_code)

                i += 1

            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[output_mod_index])
            outputs.append(refined_latent_code)

        return outputs

    def get_call_flat_latent_codes(self, inputs, fuse_with=None):
        inputs_latent_codes = self.get_flat_latent_codes(inputs)
        if fuse_with is None:
            fuse_with_latent_codes = inputs_latent_codes
        else:
            fuse_with_latent_codes = self.get_flat_latent_codes(fuse_with)
        return inputs_latent_codes, fuse_with_latent_codes

    @tf.function
    def get_flat_latent_codes(self, latent_codes):
        batch_size = tf.shape(latent_codes[0])[0]
        flat_latent_codes = []
        for i in range(self.modality_count):
            latent_code = tf.reshape(latent_codes[i], [batch_size, self.latent_code_sizes[i]])
            flat_latent_codes.append(latent_code)
        return flat_latent_codes

    @property
    def modality_count(self):
        return len(self.projection_layers)


def main():
    from protocols import Protocol, ProtocolTrainConfig, ImageCallbackConfig, AudioCallbackConfig, AUCCallbackConfig
    from protocols.utils import make_residual_encoder, make_residual_decoder
    from modalities import Pattern, ModalityLoadInfo, MelSpectrogram, RawVideo
    from preprocessing.video_preprocessing import make_video_augmentation, make_video_preprocess

    base_step_size = 16

    video_step_size = base_step_size
    video_height = 128
    video_width = 128
    video_channels = 1
    audio_step_size = base_step_size * 4
    audio_channels = 100
    step_count = 4

    # region IAEs
    video_encoder = make_residual_encoder(input_shape=(video_step_size, video_height, video_width, video_channels),
                                          filters=[32, 64, 64, 64, 128],
                                          kernel_size=3,
                                          strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), ],
                                          code_size=64, code_activation="tanh", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="VideoEncoder")

    audio_encoder = make_residual_encoder(input_shape=(audio_step_size, audio_channels),
                                          filters=[128, 128, 128, 128, 128],
                                          kernel_size=5,
                                          strides=[2, 2, 2, 2, 2],
                                          code_size=128, code_activation="tanh", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="AudioEncoder")

    video_decoder = make_residual_decoder(input_shape=video_encoder.output_shape[1:],
                                          filters=[128, 64, 64, 64, 32],
                                          kernel_size=3,
                                          strides=[(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), ],
                                          channels=video_channels, output_activation="linear", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="VideoDecoder")

    audio_decoder = make_residual_decoder(input_shape=audio_encoder.output_shape[1:],
                                          filters=[128, 128, 128, 128, 128],
                                          kernel_size=5,
                                          strides=[2, 2, 2, 2, 2],
                                          channels=audio_channels, output_activation="linear", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="AudioDecoder")
    video_iae = IAE(encoder=video_encoder,
                    decoder=video_decoder,
                    step_size=video_step_size,
                    name="VideoIAE")

    audio_iae = IAE(encoder=audio_encoder,
                    decoder=audio_decoder,
                    step_size=audio_step_size,
                    name="AudioIAE")
    # endregion

    miae = MIAE([audio_iae, video_iae])

    protocol = Protocol(model=miae, dataset_name="emoly", protocol_name="audio_video", model_name="miae")

    # region Train pattern
    video_augment = make_video_augmentation(length=video_step_size * step_count,
                                            height=video_height,
                                            width=video_width,
                                            channels=3,
                                            crop_ratio=0.2,
                                            dropout_noise_ratio=0.0,
                                            negative_prob=0.0)

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
        ModalityLoadInfo(MelSpectrogram, audio_step_size * step_count),
        ModalityLoadInfo(RawVideo, video_step_size * step_count),
        output_map=lambda audio, video: (audio, video_preprocess(video))
    )

    train_image_callback_config = ImageCallbackConfig(autoencoder=miae,
                                                      pattern=image_callback_pattern,
                                                      is_train_callback=True,
                                                      name="train",
                                                      modality_index=1)

    test_image_callback_config = ImageCallbackConfig(autoencoder=miae,
                                                     pattern=image_callback_pattern,
                                                     is_train_callback=False,
                                                     name="test",
                                                     modality_index=1)

    interpolation_image_callback_config = ImageCallbackConfig(autoencoder=miae.interpolate,
                                                              pattern=image_callback_pattern,
                                                              is_train_callback=False,
                                                              name="interpolation",
                                                              modality_index=1
                                                              )

    image_callback_configs = [train_image_callback_config, test_image_callback_config,
                              interpolation_image_callback_config]

    # endregion

    # region Audio callback config

    audio_callback_config = AudioCallbackConfig(autoencoder=miae,
                                                pattern=image_callback_pattern,
                                                is_train_callback=True,
                                                name="train",
                                                modality_index=0,
                                                mel_spectrogram=protocol.dataset_config.modalities[MelSpectrogram]
                                                )

    # endregion

    # region AUC callback config
    def preprocess_audio_video_labels(audio, video, labels):
        inputs = (audio, video_preprocess(video))
        return inputs, labels

    miae_auc_callback_config = AUCCallbackConfig(autoencoder=miae,
                                                 pattern=Pattern(
                                                     ModalityLoadInfo(MelSpectrogram, audio_step_size * step_count),
                                                     ModalityLoadInfo(RawVideo, video_step_size * step_count),
                                                     output_map=preprocess_audio_video_labels
                                                 ).with_labels(),
                                                 labels_length=step_count,
                                                 prefix="AudioVideo",
                                                 metrics=miae.modalities_mse
                                                 )

    # video_auc_callback_config = AUCCallbackConfig(autoencoder=video_iae,
    #                                               pattern=Pattern(
    #                                                   ModalityLoadInfo(RawVideo, video_step_size * step_count),
    #                                                   output_map=video_preprocess
    #                                               ).with_labels(),
    #                                               labels_length=step_count,
    #                                               prefix="VideoOnly",
    #                                               metrics=video_iae.step_mse
    #                                               )
    #
    # audio_auc_callback_config = AUCCallbackConfig(autoencoder=audio_iae,
    #                                               pattern=Pattern(
    #                                                   ModalityLoadInfo(MelSpectrogram, audio_step_size * step_count),
    #                                               ).with_labels(),
    #                                               labels_length=step_count,
    #                                               prefix="AudioOnly",
    #                                               metrics=audio_iae.step_mse
    #                                               )
    # auc_callbacks_configs = [miae_auc_callback_config, video_auc_callback_config, audio_auc_callback_config]
    auc_callbacks_configs = [miae_auc_callback_config]
    # endregion

    train_config = ProtocolTrainConfig(batch_size=8,
                                       pattern=train_pattern,
                                       steps_per_epoch=5000,
                                       epochs=1000,
                                       initial_epoch=0,
                                       validation_steps=128,
                                       image_callbacks_configs=image_callback_configs,
                                       audio_callbacks_configs=[audio_callback_config],
                                       auc_callbacks_configs=auc_callbacks_configs
                                       )

    protocol.train_model(train_config)


if __name__ == "__main__":
    main()
