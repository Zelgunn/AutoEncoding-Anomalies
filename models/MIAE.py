import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
import numpy as np
from typing import List, Dict, Tuple

from models import CustomModel, IAE


class MIAE(CustomModel):
    def __init__(self,
                 iaes: List[IAE],
                 learning_rate=1e-3):
        super(MIAE, self).__init__()

        self.iaes = iaes
        latent_code_sizes = [np.prod(iae.encoder.output_shape[1:]) for iae in iaes]
        self.fusion_model = FusionModel(latent_code_sizes)

        self.optimizer = None
        self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=None, mask=None):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.iaes[i].encode(inputs[i])
            latent_codes.append(latent_code)

        refined_latent_codes = self.fusion_model(latent_codes)

        outputs = []
        for i in range(self.modality_count):
            output = self.iaes[i].decode(refined_latent_codes[i])
            outputs.append(output)

        return outputs

    @property
    def metrics_names(self):
        return ["reconstruction"] + ["modality_{}_reconstruction".format(i) for i in range(self.modality_count)]

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
        outputs = self(inputs)
        losses = []
        for i in range(self.modality_count):
            modality_loss: tf.Tensor = tf.reduce_mean(tf.square(inputs[i] - outputs[i]))
            losses.append(modality_loss)

        return tuple(losses)

    # def compute_encoded_shape(self, input_shape):
    #     return self.encoder.compute_output_shape(input_shape)

    # def compute_output_signature(self, input_signature):
    #     return input_signature

    @property
    def modality_count(self):
        return len(self.iaes)

    def get_config(self):
        config = {iae.name: iae.get_config() for iae in self.iaes}
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {iae: iae.name for iae in self.iaes}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for iae in self.iaes:
            iae.set_optimizer(optimizer)


class FusionModel(Model):
    def __init__(self, latent_code_sizes: List[int]):
        super(FusionModel, self).__init__()

        self.latent_code_sizes = latent_code_sizes
        self.projection_layers = []
        for latent_code_size in latent_code_sizes:
            self.projection_layers.append(Dense(units=latent_code_size, activation="tanh"))

    def call(self, inputs, training=None, mask=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]
        batch_size = code_shapes[0][0]
        latent_codes = []

        for i in range(self.modality_count):
            flat_latent_code = tf.reshape(inputs[i], [batch_size, self.latent_code_sizes[i]])
            latent_codes.append(flat_latent_code)

        fused_latent_codes = tf.concat(latent_codes, axis=-1)

        outputs = []
        for i in range(self.modality_count):
            refined_latent_code = self.projection_layers[i](fused_latent_codes)
            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[i])
            outputs.append(refined_latent_code)

        return outputs

    @property
    def modality_count(self):
        return len(self.projection_layers)


def main():
    from protocols.utils import make_residual_encoder, make_residual_decoder

    video_step_size = 8
    video_height = 128
    video_width = 128
    video_channels = 1
    audio_step_size = 128
    audio_channels = 100
    step_count = 4

    video_encoder = make_residual_encoder(input_shape=(video_step_size, video_height, video_width, video_channels),
                                          filters=[4, 8, 16, 32, 64],
                                          kernel_size=3,
                                          strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), ],
                                          code_size=64, code_activation="tanh", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="VideoEncoder")

    audio_encoder = make_residual_encoder(input_shape=(audio_step_size, audio_channels),
                                          filters=[128, 128, 128, 128, 128],
                                          kernel_size=3,
                                          strides=[2, 2, 2, 2, 2],
                                          code_size=128, code_activation="tanh", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="AudioEncoder")

    video_decoder = make_residual_decoder(input_shape=video_encoder.output_shape[1:],
                                          filters=[64, 32, 16, 8, 4],
                                          kernel_size=3,
                                          strides=[(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), ],
                                          channels=video_channels, output_activation="linear", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="VideoDecoder")

    audio_decoder = make_residual_decoder(input_shape=audio_encoder.output_shape[1:],
                                          filters=[128, 128, 128, 128, 128],
                                          kernel_size=3,
                                          strides=[2, 2, 2, 2, 2],
                                          channels=audio_channels, output_activation="linear", use_batch_norm=False,
                                          use_residual_bias=False, use_conv_bias=True,
                                          name="AudioDecoder")
    video_iae = IAE(encoder=video_encoder,
                    decoder=video_decoder,
                    step_size=video_step_size)

    audio_iae = IAE(encoder=audio_encoder,
                    decoder=audio_decoder,
                    step_size=audio_step_size)

    miae = MIAE([audio_iae, video_iae])

    from protocols import Protocol, ProtocolTrainConfig
    from modalities import Pattern, ModalityLoadInfo, MelSpectrogram, RawVideo
    from preprocessing.video_preprocessing import make_video_augmentation

    protocol = Protocol(model=miae, dataset_name="emoly", protocol_name="audio_video", model_name="miae")

    video_augment = make_video_augmentation(length=video_step_size * step_count,
                                            height=video_height,
                                            width=video_width,
                                            channels=video_channels,
                                            crop_ratio=0.2,
                                            dropout_noise_ratio=0.0,
                                            negative_prob=0.0)

    def output_map(inputs):
        audio, video = inputs
        video = video_augment(video)
        return audio, video

    train_config = ProtocolTrainConfig(batch_size=8,
                                       pattern=Pattern(
                                           ModalityLoadInfo(MelSpectrogram, audio_step_size * step_count),
                                           ModalityLoadInfo(RawVideo, video_step_size * step_count),
                                           output_map=output_map
                                       ),
                                       epochs=100,
                                       initial_epoch=0
                                       )

    protocol.train_model(train_config)


if __name__ == "__main__":
    main()
