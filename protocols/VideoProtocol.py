import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
import numpy as np
import cv2
from typing import List, Tuple

from CustomKerasLayers import ConvAM
from protocols import DatasetProtocol
from callbacks.configs import ImageCallbackConfig
from protocols.utils import make_encoder, make_decoder, make_discriminator
from modalities import Pattern, ModalityLoadInfo, RawVideo
from custom_tf_models import AE, IAE
from custom_tf_models.autoregressive import SAAM, AND
from custom_tf_models.adversarial import IAEGAN, VAEGAN
from custom_tf_models.energy_based import EBGAN
from data_processing.video_preprocessing import make_video_augmentation, make_video_preprocess


# from misc_utils.train_utils import ScaledSchedule
# from misc_utils.train_utils import WarmupSchedule


class VideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 initial_epoch: int,
                 model_name: str = None
                 ):
        super(VideoProtocol, self).__init__(dataset_name=dataset_name,
                                            protocol_name="video",
                                            initial_epoch=initial_epoch,
                                            model_name=model_name)

    # region Make model
    def make_model(self) -> Model:
        if self.model_architecture == "ae":
            return self.make_ae()
        elif self.model_architecture == "iae":
            return self.make_iae()
        elif self.model_architecture == "iaegan":
            return self.make_iaegan()
        elif self.model_architecture == "vaegan":
            return self.make_vaegan()
        elif self.model_architecture == "ebgan":
            return self.make_ebgan()
        elif self.model_architecture == "and":
            return self.make_and()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

    def make_ae(self) -> AE:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])

        model = AE(encoder=encoder,
                   decoder=decoder,
                   learning_rate=self.base_learning_rate_schedule)
        return model

    def make_iae(self) -> IAE:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])

        model = IAE(encoder=encoder,
                    decoder=decoder,
                    step_size=self.step_size,
                    learning_rate=self.base_learning_rate_schedule,
                    seed=self.seed)
        return model

    def make_iaegan(self) -> IAEGAN:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])

        discriminator_input_shape = self.get_encoder_input_shape()[1:]
        discriminator = self.make_discriminator(discriminator_input_shape)

        model = IAEGAN(encoder=encoder,
                       decoder=decoder,
                       discriminator=discriminator,
                       step_size=self.step_size,
                       autoencoder_learning_rate=self.base_learning_rate_schedule,
                       discriminator_learning_rate=self.discriminator_learning_rate_schedule,
                       seed=self.seed
                       )
        return model

    def make_vaegan(self) -> VAEGAN:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])
        discriminator = self.make_discriminator(self.get_encoder_input_shape()[1:])

        model = VAEGAN(encoder=encoder,
                       decoder=decoder,
                       discriminator=discriminator,
                       learned_reconstruction_loss_factor=0.0,
                       reconstruction_loss_factor=100.0,
                       kl_divergence_loss_factor=10.0,
                       balance_discriminator_learning_rate=False,
                       seed=self.seed
                       )

        return model

    def make_ebgan(self) -> EBGAN:
        autoencoder = self.make_ae()
        generator = self.make_decoder(autoencoder.decoder.input_shape[1:], name="Generator")

        model = EBGAN(autoencoder=autoencoder,
                      generator=generator,
                      margin=1e-3,
                      seed=self.seed,
                      )
        return model

    def make_and(self) -> AND:
        encoder = self.make_encoder(self.get_encoder_input_shape()[1:])
        decoder = self.make_decoder(self.get_latent_code_shape(encoder)[1:])
        conv_am = self.make_conv_am(self.get_saam_input_shape()[1:])

        model = AND(encoder=encoder,
                    decoder=decoder,
                    am=conv_am,
                    step_size=self.step_size,
                    learning_rate=self.learning_rate)

        return model

    # endregion

    # region Sub-models input shapes
    def get_encoder_input_shape(self):
        shape = (None, self.step_size, self.height, self.width, 1)
        return shape

    def get_latent_code_shape(self, encoder: Model):
        shape = encoder.compute_output_shape(self.get_encoder_input_shape())
        if self.model_architecture in ["vaegan"]:
            shape = (*shape[:-1], shape[-1] // 2)
        return shape

    def get_saam_input_shape(self):
        shape = (None, self.step_count, self.code_size)
        return shape

    # endregion

    # region Make sub-models
    def make_encoder(self, input_shape, name="ResidualEncoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               filters=self.encoder_filters,
                               kernel_size=self.kernel_size,
                               strides=self.encoder_strides,
                               code_size=self.code_size,
                               code_activation=self.code_activation,
                               model_depth=self.model_depth,
                               seed=self.seed,
                               name=name,
                               )
        return encoder

    def make_decoder(self, input_shape, name="ResidualDecoder") -> Model:
        decoder = make_decoder(input_shape=input_shape,
                               filters=self.decoder_filters,
                               kernel_size=self.kernel_size,
                               strides=self.decoder_strides,
                               channels=1,
                               output_activation=self.output_activation,
                               model_depth=self.model_depth,
                               seed=self.seed,
                               name=name,
                               )
        return decoder

    def make_discriminator(self, input_shape) -> Model:
        discriminator_config = self.config["discriminator"]
        include_intermediate_output = self.model_architecture in ["vaegan"]
        discriminator = make_discriminator(input_shape=input_shape,
                                           filters=discriminator_config["filters"],
                                           kernel_size=self.kernel_size,
                                           strides=discriminator_config["strides"],
                                           intermediate_size=discriminator_config["intermediate_size"],
                                           intermediate_activation="relu",
                                           include_intermediate_output=include_intermediate_output,
                                           seed=self.seed)
        return discriminator

    def make_saam(self, input_shape) -> SAAM:
        saam_config = self.config["saam"]
        saam = SAAM(layer_count=saam_config["layer_count"],
                    head_count=saam_config["layer_count"],
                    head_size=saam_config["layer_count"],
                    intermediate_size=saam_config["layer_count"],
                    output_size=saam_config["layer_count"],
                    output_activation="softmax",
                    input_shape=input_shape)
        return saam

    def make_conv_am(self, input_shape):
        conv_am = ConvAM(rank=2,
                         filters=self.config["conv_am"]["filters"],
                         intermediate_activation="relu",
                         output_activation="linear",
                         input_shape=input_shape)
        return conv_am

    # endregion

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        augment_video = self.make_video_augmentation()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            preprocessor=augment_video
        )
        return pattern

    def get_image_pattern(self) -> Pattern:
        video_preprocess = self.make_video_preprocess()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            preprocessor=video_preprocess
        )
        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_image_pattern().with_labels()

    # endregion

    # region Pre-processes
    def make_video_augmentation(self):
        negative_prob = 0.5 if self.use_random_negative else 0.0
        return make_video_augmentation(length=self.output_length,
                                       height=self.height,
                                       width=self.width,
                                       channels=self.dataset_channels,
                                       crop_ratio=self.crop_ratio,
                                       dropout_noise_ratio=self.dropout_noise_ratio,
                                       negative_prob=negative_prob,
                                       seed=self.seed)

    def make_video_preprocess(self):
        return make_video_preprocess(height=self.height,
                                     width=self.width,
                                     to_grayscale=self.dataset_channels == 3)

    # endregion

    def get_modality_callback_configs(self):
        image_pattern = self.get_image_pattern()

        image_callbacks_configs = [
            ImageCallbackConfig(autoencoder=self.model, pattern=image_pattern, is_train_callback=True,
                                name="train", video_sample_rate=self.video_sample_rate),
            ImageCallbackConfig(autoencoder=self.model, pattern=image_pattern, is_train_callback=False,
                                name="test", video_sample_rate=self.video_sample_rate),
        ]

        if isinstance(self.model, IAE):
            image_callback_config = ImageCallbackConfig(autoencoder=self.model.interpolate, pattern=image_pattern,
                                                        is_train_callback=False, name="interpolate_test",
                                                        video_sample_rate=self.video_sample_rate)
            image_callbacks_configs.append(image_callback_config)

        return image_callbacks_configs

    # region Properties
    # region Abstract properties
    @property
    @abstractmethod
    def dataset_channels(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def use_face(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def video_sample_rate(self) -> int:
        raise NotImplementedError

    # endregion
    # region Config properties
    @property
    def model_architecture(self) -> str:
        return self.config["model_architecture"].lower()

    @property
    def height(self) -> int:
        return self.config["height"]

    @property
    def width(self) -> int:
        return self.config["width"]

    @property
    def step_size(self) -> int:
        return self.config["step_size"]

    @property
    def step_count(self) -> int:
        return self.config["step_count"]

    @property
    def code_size(self) -> int:
        return self.config["code_size"]

    # region Encoder
    @property
    def encoder_config(self):
        return self.config["encoder"]

    @property
    def encoder_filters(self) -> List[int]:
        return self.encoder_config["filters"]

    @property
    def encoder_strides(self) -> List[List[int]]:
        return self.encoder_config["strides"]

    @property
    def code_activation(self) -> str:
        return self.config["code_activation"]

    # endregion

    # region Decoder
    @property
    def decoder_config(self):
        return self.config["decoder"]

    @property
    def decoder_filters(self) -> List[int]:
        return self.decoder_config["filters"]

    @property
    def decoder_strides(self) -> List[List[int]]:
        return self.decoder_config["strides"]

    # endregion

    @property
    def model_depth(self) -> int:
        return len(self.encoder_filters) + len(self.decoder_filters)

    @property
    def kernel_size(self) -> int:
        return self.config["kernel_size"]

    @property
    def use_batch_norm(self) -> bool:
        return self.config["use_batch_norm"]

    @property
    def learning_rate(self) -> float:
        return self.config["learning_rate"]

    @property
    def base_learning_rate_schedule(self):
        learning_rate = self.learning_rate
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 2000, 0.8, staircase=False)
        # learning_rate = WarmupSchedule(warmup_steps=1000, learning_rate=learning_rate)
        return learning_rate

    @property
    def discriminator_learning_rate_schedule(self):
        learning_rate = self.base_learning_rate_schedule
        # learning_rate = ScaledSchedule(learning_rate, 1.0)
        return learning_rate

    # region Data augmentation
    @property
    def data_augmentation_config(self):
        return self.config["data_augmentation"]

    @property
    def use_cropping(self) -> bool:
        return self.crop_ratio > 0.0

    @property
    def crop_ratio(self) -> float:
        return self.data_augmentation_config["crop_ratio"]

    @property
    def dropout_noise_ratio(self) -> float:
        return self.data_augmentation_config["dropout_noise_ratio"]

    @property
    def gaussian_noise_ratio(self) -> float:
        return self.data_augmentation_config["gaussian_noise_ratio"]

    @property
    def use_random_negative(self) -> bool:
        return self.data_augmentation_config["use_random_negative"]

    # endregion

    # endregion

    @property
    def output_length(self) -> int:
        if self.model_architecture in ["iae", "and", "iaegan"]:
            return self.step_size * self.step_count
        else:
            return self.step_size

    # endregion

    def autoencode_video(self, video_source, target_path: str, load_epoch: int, fps=25.0):
        from datasets.data_readers import VideoReader
        from custom_tf_models.utils import reduce_std_from, reduce_mean_from
        from tqdm import tqdm

        self.load_weights(epoch=load_epoch)

        video_reader = VideoReader(video_source)
        output_size = (self.width, self.height)
        video_writer = cv2.VideoWriter(target_path, cv2.VideoWriter.fourcc(*"H264"), fps, output_size)

        frames = []
        i = 0
        for frame in tqdm(video_reader, total=video_reader.frame_count):
            frames.append(frame)
            i += 1

            if len(frames) == self.output_length:
                inputs = np.stack(frames, axis=0)

                if inputs.ndim == 3:
                    inputs = np.expand_dims(inputs, axis=-1)

                inputs = tf.image.resize(inputs, output_size)

                std = reduce_std_from(inputs, keepdims=True)
                mean = reduce_mean_from(inputs, keepdims=True)
                inputs = (inputs - mean) / std
                inputs = tf.expand_dims(inputs, axis=0)

                outputs = self.autoencoder(inputs)
                outputs = tf.squeeze(outputs, axis=0)
                outputs = outputs * std + mean
                outputs = outputs.numpy()

                if i < video_reader.frame_count:
                    write_frame(video_writer, outputs[0], output_size)
                else:
                    for output in outputs:
                        write_frame(video_writer, output, output_size)

                frames.pop(0)

        video_writer.release()

    def process_frame(self, frame):
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32)
            frame /= 255

        frame = cv2.resize(frame, dsize=(self.width, self.height))

        if frame.ndim == 2:
            frame = np.expand_dims(frame, axis=-1)
        else:
            frame = np.mean(frame, axis=-1, keepdims=True)

        return frame


def write_frame(video_writer: cv2.VideoWriter, frame: np.ndarray, output_size: Tuple[int, int]):
    frame = cv2.resize(frame, output_size)
    frame = frame.astype(np.uint8)

    if frame.ndim == 2:
        frame = np.expand_dims(frame, axis=-1)
        frame = np.tile(frame, reps=[1, 1, 3])

    video_writer.write(frame)
