import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
from typing import List

from protocols import DatasetProtocol
from protocols import ImageCallbackConfig
from protocols.utils import make_residual_encoder, make_residual_decoder
from modalities import Pattern, ModalityLoadInfo, RawVideo
from models import IAE


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

    def make_model(self) -> Model:
        if self.model_architecture is "iae":
            return self.make_iae()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

    def make_iae(self) -> IAE:
        encoder_input_shape = (None, self.step_size, self.height, self.width, 1)
        encoder = self.make_encoder(encoder_input_shape[1:])

        decoder_input_shape = encoder.compute_output_shape(encoder_input_shape)
        decoder = self.make_decoder(decoder_input_shape[1:])

        model = IAE(encoder=encoder,
                    decoder=decoder,
                    step_size=self.step_size)
        return model

    def make_encoder(self, input_shape):
        encoder = make_residual_encoder(input_shape=input_shape,
                                        filters=self.encoder_filters,
                                        strides=self.encoder_strides,
                                        code_size=self.code_size,
                                        code_activation=self.code_activation,)
        return encoder

    def make_decoder(self, input_shape):
        decoder = make_residual_decoder(input_shape=input_shape,
                                        filters=self.decoder_filters,
                                        strides=self.decoder_strides,
                                        channels=1,
                                        output_activation=self.output_activation)
        return decoder

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        augment_video = self.make_video_augmentation()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            output_map=augment_video
        )
        return pattern

    def get_image_pattern(self) -> Pattern:
        video_preprocess = self.make_video_preprocess()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            output_map=video_preprocess
        )
        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        return self.get_image_pattern().with_labels()

    # endregion

    # region Pre-processes
    def make_video_augmentation(self):
        preprocess_video = self.make_video_preprocess()

        def augment_video(video: tf.Tensor) -> tf.Tensor:
            if self.use_cropping:
                crop_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.0)
                original_shape = tf.cast(tf.shape(video), tf.float32)
                original_height, original_width = original_shape[1], original_shape[2]
                crop_size = [self.output_length, crop_ratio * original_height, crop_ratio * original_width, 1]
                video = tf.image.random_crop(video, crop_size)

            video = preprocess_video(video)
            return video

        return augment_video

    def make_video_preprocess(self):
        def preprocess(video: tf.Tensor, labels: tf.Tensor = None):
            video = tf.image.resize(video, (self.height, self.width))

            if self.dataset_channels == 3:
                rgb_weights = [0.2989, 0.5870, 0.1140]
                rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
                video *= rgb_weights
                video = tf.reduce_sum(video, axis=-1, keepdims=True)

            if labels is None:
                return video
            else:
                return video, labels

        return preprocess

    # endregion

    def get_image_callback_configs(self):
        image_pattern = self.get_image_pattern()

        image_callbacks_configs = [
            ImageCallbackConfig(self.model, image_pattern, True, "train"),
            ImageCallbackConfig(self.model, image_pattern, False, "test"),
        ]

        if isinstance(self.model, IAE):
            image_callbacks_configs += \
                [ImageCallbackConfig(self.model.interpolate, image_pattern, False, "interpolate_test")]

        return image_callbacks_configs

    # region Properties
    # region Abstract properties
    @property
    @abstractmethod
    def dataset_channels(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def use_cropping(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def use_face(self) -> bool:
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

    @property
    def encoder_filters(self) -> List[int]:
        return self.config["encoder_filters"]

    @property
    def encoder_strides(self) -> List[List[int]]:
        return self.config["encoder_strides"]

    @property
    def code_activation(self) -> str:
        return self.config["code_activation"]

    @property
    def decoder_filters(self) -> List[int]:
        return self.config["decoder_filters"]

    @property
    def decoder_strides(self) -> List[List[int]]:
        return self.config["decoder_strides"]

    @property
    def output_activation(self) -> str:
        return self.config["output_activation"]
    # endregion

    @property
    def output_length(self) -> int:
        if self.model_architecture is "iae":
            return self.step_size * self.step_count
        else:
            return self.step_size
    # endregion
