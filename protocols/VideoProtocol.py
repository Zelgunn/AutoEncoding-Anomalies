import tensorflow as tf

from protocols import DatasetProtocol
from protocols import ImageCallbackConfig
from modalities import Pattern, ModalityLoadInfo, RawVideo
from models import IAE


# noinspection PyAbstractClass
class VideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 height: int,
                 width: int,
                 model_name: str = None
                 ):
        self.height = height
        self.width = width

        super(VideoProtocol, self).__init__(dataset_name=dataset_name,
                                            protocol_name=protocol_name,
                                            model_name=model_name)

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
