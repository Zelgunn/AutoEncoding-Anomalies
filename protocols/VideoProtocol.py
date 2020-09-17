import tensorflow as tf
from tensorflow.python.keras import Model, optimizers
from abc import abstractmethod
import numpy as np
import cv2
from typing import List, Tuple, Callable, Union

from CustomKerasLayers import ConvAM
from protocols import DatasetProtocol
from callbacks.configs import ImageCallbackConfig
from protocols.utils import make_encoder, make_decoder, make_discriminator
from modalities import Pattern, ModalityLoadInfo, RawVideo
from custom_tf_models import AE, IAE, BinAE
from custom_tf_models import LED, ALED
from custom_tf_models.autoregressive import SAAM, AND
from custom_tf_models.adversarial import IAEGAN, VAEGAN
from custom_tf_models.energy_based import EBGAN, EBM, EBAE
from data_processing.video_preprocessing import make_video_augmentation, make_video_preprocess

from data_processing.video_processing.VideoPatchExtractor import VideoPatchExtractor


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
            model = self.make_ae()
        elif self.model_architecture == "iae":
            model = self.make_iae()
        elif self.model_architecture == "bin_ae":
            model = self.make_bin_ae()
        elif self.model_architecture == "iaegan":
            model = self.make_iaegan()
        elif self.model_architecture == "vaegan":
            model = self.make_vaegan()
        elif self.model_architecture == "and":
            model = self.make_and()
        elif self.model_architecture == "led":
            model = self.make_led()
        elif self.model_architecture == "aled":
            model = self.make_aled()
        elif self.model_architecture == "ebm":
            model = self.make_ebm()
        elif self.model_architecture == "ebae":
            model = self.make_ebae()
        elif self.model_architecture == "ebgan":
            model = self.make_ebgan()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

        self.setup_model(model)
        return model

    def setup_model(self, model: Model):
        model.build(self.get_encoder_input_batch_shape(False))
        if isinstance(model, (IAE, LED)):
            # noinspection PyProtectedMember
            model._set_inputs(tf.zeros(self.get_encoder_input_batch_shape(True)))
        model.compile(optimizer=self.make_base_optimizer())

    # region AE
    def make_ae(self) -> AE:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = AE(encoder=encoder, decoder=decoder)
        return model

    def make_bin_ae(self) -> BinAE:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = BinAE(encoder=encoder, decoder=decoder)
        return model

    # endregion

    # region IAE
    def make_iae(self) -> IAE:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = IAE(encoder=encoder, decoder=decoder, step_size=self.step_size, seed=self.seed)
        return model

    def make_iaegan(self) -> IAEGAN:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        discriminator_input_shape = self.get_encoder_input_shape()
        discriminator = self.make_discriminator(discriminator_input_shape)

        model = IAEGAN(encoder=encoder,
                       decoder=decoder,
                       discriminator=discriminator,
                       step_size=self.step_size,
                       discriminator_optimizer=self.make_discriminator_optimizer(),
                       seed=self.seed
                       )
        return model

    # endregion

    # region VAE
    def make_vaegan(self) -> VAEGAN:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))
        discriminator = self.make_discriminator(self.get_encoder_input_shape())

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

    # endregion

    # region Autoregressive
    def make_and(self) -> AND:
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))
        conv_am = self.make_conv_am(self.get_saam_input_shape()[1:])

        model = AND(encoder=encoder, decoder=decoder, am=conv_am, step_size=self.step_size)

        return model

    # endregion

    # region Minimalist Desc
    def make_led(self):
        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = LED(encoder=encoder,
                    decoder=decoder,
                    features_per_block=1,
                    merge_dims_with_features=False,
                    add_binarization_noise_to_mask=True,
                    description_energy_loss_lambda=1e-3,
                    noise_stddev=0.25,
                    reconstruct_noise=True,
                    seed=self.seed)
        return model

    def make_aled(self):
        encoder = self.make_encoder(self.get_encoder_input_shape())
        latent_code_shape = self.get_latent_code_shape(encoder)
        decoder = self.make_decoder(latent_code_shape)
        generator = self.make_decoder(latent_code_shape, name="ResidualGenerator")
        # generator_learning_rate = self.base_learning_rate_schedule

        model = ALED(encoder=encoder,
                     decoder=decoder,
                     generator=generator,
                     # generator_learning_rate=generator_learning_rate,
                     features_per_block=1,
                     merge_dims_with_features=False,
                     add_binarization_noise_to_mask=True,
                     noise_stddev=0.25,
                     reconstruct_noise=False,
                     seed=self.seed)
        return model

    # endregion

    # region Energy based
    def make_ebm(self):
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Flatten
        from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling
        from custom_tf_models.energy_based.energy_state_functions.FlipSequence import FlipSequence
        from custom_tf_models.energy_based.energy_state_functions.IdentityESF import IdentityESF

        encoder = self.make_encoder(self.get_encoder_input_shape())
        kernel_initializer = VarianceScaling(seed=self.seed)
        energy_model = Sequential(
            layers=[
                encoder,
                Flatten(),
                Dense(units=64, activation="relu", kernel_initializer=kernel_initializer),
                Dense(units=1, activation="linear", kernel_initializer=kernel_initializer, use_bias=False),
            ],
            name="EnergyModel"
        )

        model = EBM(energy_model=energy_model,
                    energy_state_functions=[FlipSequence(), IdentityESF()],
                    optimizer=tf.keras.optimizers.Adam(),
                    energy_margin=1.0,
                    seed=self.seed)

        return model

    def make_ebae(self) -> EBAE:
        from custom_tf_models.energy_based.energy_state_functions.FlipSequence import FlipSequence
        from custom_tf_models.energy_based.energy_state_functions.IdentityESF import IdentityESF

        encoder = self.make_encoder(self.get_encoder_input_shape())
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        model = EBAE(encoder=encoder,
                     decoder=decoder,
                     energy_state_functions=[FlipSequence(), IdentityESF()],
                     energy_margin=2e-1,
                     weights_decay=0.0,
                     seed=self.seed)

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

    # endregion

    # endregion

    # region Sub-models input shapes
    def get_encoder_input_batch_shape(self, use_batch_size=False) -> Tuple[None, int, int, int, int]:
        batch_size = self.batch_size if use_batch_size else None
        shape = (batch_size, self.step_size, self.height, self.width, 1)
        return shape

    def get_encoder_input_shape(self) -> Tuple[int, int, int, int]:
        shape = self.get_encoder_input_batch_shape()[1:]
        return shape

    def get_latent_code_batch_shape(self, encoder: Model):
        shape = encoder.compute_output_shape(self.get_encoder_input_batch_shape())
        if self.model_architecture in ["vaegan"]:
            shape = (*shape[:-1], shape[-1] // 2)
        return shape

    def get_latent_code_shape(self, encoder: Model):
        return self.get_latent_code_batch_shape(encoder=encoder)[1:]

    def get_saam_input_shape(self):
        shape = (None, self.step_count, self.code_size)
        return shape

    # endregion

    # region Make sub-models
    # region Base
    def make_encoder(self, input_shape, name="ResidualEncoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               filters=self.encoder_filters,
                               kernel_size=self.kernel_size,
                               strides=self.encoder_strides,
                               code_size=self.code_size,
                               code_activation=self.code_activation,
                               basic_block_count=self.basic_block_count,
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
                               basic_block_count=self.basic_block_count,
                               seed=self.seed,
                               name=name,
                               )
        return decoder

    # endregion

    # region Adversarial
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

    # endregion

    # region Autoregressive
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

    # region Optimizers
    def make_optimizer(self,
                       learning_rate: Union[Callable, float],
                       optimizer_class: str = None
                       ) -> optimizers.optimizer_v2.OptimizerV2:

        optimizer_class = self.optimizer_class if optimizer_class is None else optimizer_class
        if optimizer_class == "adam":
            return tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_class == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate)
        elif optimizer_class == "sgd":
            return tf.keras.optimizers.SGD(learning_rate)
        else:
            raise ValueError("`{}` is not a valid optimizer identifier.".format(optimizer_class))

    def make_base_optimizer(self) -> optimizers.optimizer_v2.OptimizerV2:
        return self.make_optimizer(self.base_learning_rate_schedule)

    def make_discriminator_optimizer(self) -> optimizers.optimizer_v2.OptimizerV2:
        if "discriminator_optimizer" in self.config:
            discriminator_optimizer_class = self.config["discriminator_optimizer"]
        else:
            discriminator_optimizer_class = self.optimizer_class
        return self.make_optimizer(self.discriminator_learning_rate_schedule, discriminator_optimizer_class)

    # endregion

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

        if self.extract_patches:
            video_patch_extractor = VideoPatchExtractor(patch_size=self.height)
            batch_processor = video_patch_extractor.batch_process
            postprocessor = video_patch_extractor.post_process
        else:
            batch_processor = postprocessor = None

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            preprocessor=video_preprocess,
            batch_processor=batch_processor,
            postprocessor=postprocessor,
        )
        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_image_pattern().with_labels()

    # endregion

    # region Pre-processes / Post-process
    def make_video_augmentation(self) -> Callable:
        negative_prob = 0.5 if self.use_random_negative else 0.0
        return make_video_augmentation(length=self.output_length,
                                       height=self.height,
                                       width=self.width,
                                       channels=self.dataset_channels,
                                       dropout_noise_ratio=self.dropout_noise_ratio,
                                       negative_prob=negative_prob,
                                       seed=self.seed,
                                       activation_range=self.output_activation)

    def make_video_preprocess(self) -> Callable:
        target_size = (self.height, self.width) if not self.extract_patches == "resize" else None
        return make_video_preprocess(to_grayscale=self.dataset_channels == 3,
                                     activation_range=self.output_activation,
                                     target_size=target_size)

    def make_video_post_process(self) -> Callable:
        pass

    # endregion

    # region Callbacks
    def get_modality_callback_configs(self):
        image_pattern = self.get_image_pattern()
        image_callbacks_configs = []

        model = self.model
        if isinstance(model, EBAE):
            model = model.autoencoder

        if isinstance(model, AE):
            image_callbacks_configs += [
                ImageCallbackConfig(autoencoder=model, pattern=image_pattern, is_train_callback=True,
                                    name="train", video_sample_rate=self.video_sample_rate),
                ImageCallbackConfig(autoencoder=model, pattern=image_pattern, is_train_callback=False,
                                    name="test", video_sample_rate=self.video_sample_rate),
            ]

        if isinstance(model, IAE):
            image_callback_config = ImageCallbackConfig(autoencoder=model.interpolate, pattern=image_pattern,
                                                        is_train_callback=False, name="interpolate_test",
                                                        video_sample_rate=self.video_sample_rate)
            image_callbacks_configs.append(image_callback_config)

        return image_callbacks_configs

    # endregion

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
    def kernel_size(self) -> int:
        return self.config["kernel_size"]

    @property
    def basic_block_count(self) -> int:
        return self.config["basic_block_count"]

    @property
    def optimizer_class(self) -> str:
        return self.config["optimizer"].lower()

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

    @property
    def extract_patches(self) -> bool:
        return self.config["extract_patches"]

    # endregion

    @property
    def output_length(self) -> int:
        if self.model_architecture in ["iae", "and", "iaegan"]:
            return self.step_size * self.step_count
        else:
            return self.step_size

    # endregion

    # region Misc.
    def autoencode_video(self,
                         video_source,
                         target_path: str,
                         load_epoch: int,
                         fps=25.0,
                         output_size: Tuple[int, int] = None
                         ):
        from datasets.data_readers import VideoReader
        from misc_utils.math_utils import reduce_std_from, reduce_mean_from
        from tqdm import tqdm

        self.load_weights(epoch=load_epoch)

        video_reader = VideoReader(video_source)
        output_size = output_size if output_size is not None else (self.width, self.height)
        video_writer = cv2.VideoWriter(target_path, cv2.VideoWriter.fourcc(*"H264"), fps, output_size)

        pattern = self.get_image_pattern()

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

                inputs = pattern.process_batch(inputs)

                outputs = self.autoencoder(inputs)

                if pattern.postprocessor is not None:
                    outputs = pattern.postprocessor(outputs)

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
    # endregion


def write_frame(video_writer: cv2.VideoWriter, frame: np.ndarray, output_size: Tuple[int, int]):
    frame = cv2.resize(frame, output_size)
    frame = frame.astype(np.uint8)

    if frame.ndim == 2:
        frame = np.expand_dims(frame, axis=-1)
        frame = np.tile(frame, reps=[1, 1, 3])

    video_writer.write(frame)
