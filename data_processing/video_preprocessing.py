import tensorflow as tf
from typing import Tuple, Callable, Union, Optional

from data_processing.common import dropout_noise, ActivationRange, apply_activation_range


def make_video_augmentation(length: int, height: int, width: int, channels: int,
                            activation_range: Union[ActivationRange, str],
                            dropout_noise_ratio=0.0,
                            gaussian_noise_std=0.0,
                            negative_prob=0.0,
                            ):
    to_grayscale = channels == 3

    def augment_video(video: tf.Tensor) -> tf.Tensor:
        video = tf.image.random_crop(video, size=(length, height, width, channels))

        if dropout_noise_ratio > 0.0:
            video = video_dropout_noise(video, dropout_noise_ratio, spatial_prob=0.1)

        if negative_prob > 0.0:
            video = random_image_negative(video, negative_prob=negative_prob)

        if to_grayscale:
            video = convert_to_grayscale(video)

        video = apply_activation_range(video, activation_range)

        if gaussian_noise_std > 0.0:
            video_shape = tf.shape(video)
            video += tf.random.normal(video_shape, stddev=gaussian_noise_std)

        return video

    return augment_video


def make_video_preprocess(to_grayscale: bool,
                          activation_range: Union[ActivationRange, str],
                          target_size: Tuple[int, int] = None
                          ) -> Callable[[tf.Tensor, Optional[tf.Tensor]], Tuple[tf.Tensor, Optional[tf.Tensor]]]:
    def preprocess(video: tf.Tensor, labels: tf.Tensor = None):
        if target_size is not None:
            (height, width) = target_size
            video = tf.image.resize(video, (height, width))

        if to_grayscale:
            video = convert_to_grayscale(video)

        video = apply_activation_range(video, activation_range)

        if labels is None:
            return video
        else:
            return video, labels

    return preprocess


def video_random_cropping(video: tf.Tensor, crop_ratio: float, output_length: int):
    crop_ratio = tf.random.uniform(shape=(), minval=1.0 - crop_ratio, maxval=1.0)
    original_shape = tf.cast(tf.shape(video), tf.float32)
    original_height, original_width = original_shape[1], original_shape[2]

    crop_size = [output_length, crop_ratio * original_height, crop_ratio * original_width, 1]

    video = tf.image.random_crop(video, crop_size)
    return video


def video_dropout_noise(video, max_rate, spatial_prob):
    drop_width = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5
    drop_width = tf.cast(drop_width, tf.float32)

    noise_shape_prob = [spatial_prob * (1.0 - drop_width), spatial_prob * drop_width, 1.0]
    noise_shape_prob = [0.0] * (len(video.shape) - 3) + noise_shape_prob

    video = dropout_noise(video, max_rate, noise_shape_prob)
    return video


def random_image_negative(image, negative_prob=0.5):
    negative = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32) < negative_prob
    image = tf.cond(pred=negative,
                    true_fn=lambda: 1.0 - image,
                    false_fn=lambda: image)
    return image


def convert_to_grayscale(images: tf.Tensor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
    images *= rgb_weights
    images = tf.reduce_sum(images, axis=-1, keepdims=True)
    return images
