import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
import numpy as np
import cv2
from typing import Tuple, Callable, Optional, Dict

from protocols import DatasetProtocol
from callbacks.configs import ImageCallbackConfig
from modalities import Pattern, ModalityLoadInfo, RawVideo
from custom_tf_models import AE, IAE, BinAE, AEP
from custom_tf_models import LED, RDL, ALED, PreLED
from custom_tf_models.adversarial import VAEGAN, AVP
from custom_tf_models.energy_based import EBGAN, EBM, EBAE
from custom_tf_models.description_length.LED import LEDGoal
from data_processing.video_processing import make_video_augmented_preprocessor, make_video_preprocessor
from data_processing.video_processing.VideoPatchExtractor import VideoPatchExtractor


class VideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 base_log_dir: str,
                 epoch: int,
                 config: Dict = None,
                 ):
        super(VideoProtocol, self).__init__(dataset_name=dataset_name,
                                            protocol_name="video",
                                            base_log_dir=base_log_dir,
                                            epoch=epoch,
                                            config=config)

    # region Make model
    def make_model(self) -> Model:
        if self.model_architecture == "ae":
            model = self.make_ae()
        elif self.model_architecture == "aep":
            model = self.make_aep()
        elif self.model_architecture == "iae":
            model = self.make_iae()
        elif self.model_architecture == "bin_ae":
            model = self.make_bin_ae()
        elif self.model_architecture == "iaegan":
            model = self.make_iaegan()
        elif self.model_architecture == "vaegan":
            model = self.make_vaegan()
        elif self.model_architecture == "avp":
            model = self.make_avp()
        elif self.model_architecture == "led":
            model = self.make_led()
        elif self.model_architecture == "rdl":
            model = self.make_rdl()
        elif self.model_architecture == "aled":
            model = self.make_aled()
        elif self.model_architecture == "preled":
            model = self.make_preled()
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

    # region AE

    def make_aep(self) -> AEP:
        encoder = self.make_encoder(self.encoder_input_shape)
        latent_code_shape = self.get_latent_code_shape(encoder)
        decoder = self.make_decoder(latent_code_shape)
        predictor = self.make_decoder(latent_code_shape)

        model = AEP(encoder=encoder,
                    decoder=decoder,
                    predictor=predictor,
                    input_length=self.step_size,
                    use_temporal_loss=False)
        return model

    def make_bin_ae(self) -> BinAE:
        encoder, decoder = self.make_encoder_decoder()

        model = BinAE(encoder=encoder, decoder=decoder)
        return model

    # endregion

    # region VAE
    def make_vaegan(self) -> VAEGAN:
        encoder, decoder = self.make_encoder_decoder()
        discriminator = self.make_discriminator(self.encoder_input_shape)

        model = VAEGAN(encoder=encoder,
                       decoder=decoder,
                       discriminator=discriminator,
                       learned_reconstruction_loss_factor=0.0,
                       reconstruction_loss_factor=100.0,
                       kl_divergence_loss_factor=10.0,
                       balance_discriminator_learning_rate=False,
                       )

        return model

    def make_avp(self) -> AVP:
        encoder_input_shape = self.encoder_input_shape
        encoder = self.make_encoder(encoder_input_shape)
        decoder = self.make_decoder(self.get_latent_code_shape(encoder))

        encoder_input_length = encoder_input_shape[0]
        discriminator_input_shape = (encoder_input_length * 2, *encoder_input_shape[1:])
        discriminator = self.make_discriminator(discriminator_input_shape)

        model = AVP(encoder=encoder,
                    decoder=decoder,
                    discriminator=discriminator,
                    prediction_lambda=1e0,
                    gradient_difference_lambda=1e-1,
                    high_level_prediction_lambda=1e-1,
                    kl_divergence_lambda=1e-3,
                    adversarial_lambda=1e-1,
                    gradient_penalty_lambda=1e1,
                    weight_decay_lambda=1e-5,
                    input_length=self.step_size)
        return model

    # endregion

    # region LED
    def make_led(self):
        encoder, decoder = self.make_encoder_decoder()

        # autoencoder = AE(encoder, decoder)
        # autoencoder.load_weights("../logs/AEA/video/ped2/weights_019")

        # encoder.trainable = decoder.trainable = False

        model = LED(encoder=encoder,
                    decoder=decoder,
                    features_per_block=1,
                    merge_dims_with_features=False,
                    description_energy_loss_lambda=1e-2,
                    use_noise=True,
                    noise_stddev=0.02,
                    reconstruct_noise=False,
                    goal_schedule=self.led_goal,
                    allow_negative_description_loss_weight=True,
                    unmasked_reconstruction_weight=1e-0,
                    )
        return model

    # region LED Variants
    def make_rdl(self):
        encoder, decoder = self.make_encoder_decoder()

        model = RDL(encoder=encoder,
                    decoder=decoder,
                    use_noise=True,
                    noise_stddev=0.1,
                    reconstruct_noise=False)
        return model

    def make_aled(self):
        encoder = self.make_encoder(self.encoder_input_shape)
        latent_code_shape = self.get_latent_code_shape(encoder)
        decoder = self.make_decoder(latent_code_shape)
        generator = self.make_decoder(latent_code_shape, name="ResidualGenerator")

        model = ALED(encoder=encoder,
                     decoder=decoder,
                     generator=generator,
                     features_per_block=1,
                     merge_dims_with_features=False,
                     add_binarization_noise_to_mask=True,
                     description_energy_loss_lambda=1e-2)
        return model

    def make_preled(self):
        encoder, decoder = self.make_encoder_decoder()
        predictor = self.make_decoder(self.get_latent_code_shape(encoder))

        model = PreLED(encoder=encoder,
                       decoder=decoder,
                       predictor=predictor,
                       input_length=self.step_size,
                       use_temporal_reconstruction_loss=False,
                       features_per_block=1,
                       merge_dims_with_features=False,
                       description_energy_loss_lambda=1e-2,
                       use_noise=True,
                       noise_stddev=0.02,
                       reconstruct_noise=False, )
        return model

    # endregion

    # endregion

    # region Energy based
    def make_ebm(self):
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Flatten
        from tensorflow.python.ops.init_ops import VarianceScaling
        from custom_tf_models.energy_based.energy_state_functions.FlipSequence import FlipSequence
        from custom_tf_models.energy_based.energy_state_functions.IdentityESF import IdentityESF

        encoder = self.make_encoder(self.encoder_input_shape)
        kernel_initializer = VarianceScaling()
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
                    energy_margin=1.0)

        return model

    def make_ebae(self) -> EBAE:
        from custom_tf_models.energy_based.energy_state_functions.FlipSequence import FlipSequence
        from custom_tf_models.energy_based.energy_state_functions.IdentityESF import IdentityESF

        encoder, decoder = self.make_encoder_decoder()

        model = EBAE(encoder=encoder,
                     decoder=decoder,
                     energy_state_functions=[FlipSequence(), IdentityESF()],
                     energy_margin=2e-1,
                     weight_decay=0.0)

        return model

    def make_ebgan(self) -> EBGAN:
        autoencoder = self.make_ae()
        generator = self.make_decoder(autoencoder.decoder.input_shape[1:], name="Generator")

        model = EBGAN(autoencoder=autoencoder,
                      generator=generator,
                      margin=1e-3)
        return model

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

    def get_image_pattern(self, include_labels: bool = False) -> Pattern:
        video_preprocess = self.make_video_preprocess(include_labels)

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

        if include_labels:
            return pattern.with_labels()

        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_image_pattern(include_labels=True)

    # endregion

    # region Pre-processes / Post-process
    def make_video_augmentation(self) -> Callable:
        negative_prob = 0.5 if self.use_random_negative else 0.0
        return make_video_augmented_preprocessor(length=self.output_length,
                                                 height=self.height,
                                                 width=self.width,
                                                 channels=self.dataset_channels,
                                                 dropout_noise_ratio=self.dropout_noise_ratio,
                                                 negative_prob=negative_prob,
                                                 activation_range=self.output_activation)

    def make_video_preprocess(self, include_labels: bool) -> Callable:
        target_size = (self.height, self.width) if not self.extract_patches == "resize" else None
        return make_video_preprocessor(to_grayscale=self.dataset_channels == 3,
                                       activation_range=self.output_activation,
                                       include_labels=include_labels,
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

        if isinstance(model, (AE, AVP)):
            image_callbacks_configs += [
                ImageCallbackConfig(autoencoder=model, pattern=image_pattern, is_train_callback=True, name="train",
                                    video_sample_rate=self.video_sample_rate),
                ImageCallbackConfig(autoencoder=model, pattern=image_pattern, is_train_callback=False, name="test",
                                    video_sample_rate=self.video_sample_rate),
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
    def height(self) -> int:
        return self.config["height"]

    @property
    def width(self) -> int:
        return self.config["width"]

    @property
    def channels(self) -> int:
        return 1

    @property
    def encoder_input_shape(self) -> Tuple[int, int, int, int]:
        return self.step_size, self.height, self.width, self.channels

    # region Data augmentation
    @property
    def data_augmentation_config(self):
        return self.config["data_augmentation"]

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
    def led_goal(self) -> Optional[LEDGoal]:
        goal_config = self.config["led"]["goal"]
        if goal_config is None:
            return None

        led_goal = LEDGoal(initial_rate=goal_config["initial_rate"],
                           decay_steps=goal_config["decay_steps"],
                           decay_rate=goal_config["decay_rate"],
                           staircase=goal_config["staircase"],
                           offset=goal_config["offset"])
        return led_goal

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
