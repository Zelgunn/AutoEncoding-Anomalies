import tensorflow as tf

from preprocessing.common import dropout_noise


def make_video_augmentation(length: int, height: int, width: int, channels: int,
                            crop_ratio=0.0,
                            dropout_noise_ratio=0.0,
                            negative_prob=0.0,
                            ):
    preprocess_video = make_video_preprocess(height=height, width=width, base_channels=channels)

    def augment_video(video: tf.Tensor) -> tf.Tensor:
        if crop_ratio > 0.0:
            video = video_random_cropping(video, crop_ratio, length)

        if dropout_noise_ratio > 0.0:
            video = video_dropout_noise(video, dropout_noise_ratio, spatial_prob=0.1)

        if negative_prob > 0.0:
            video = random_image_negative(video, negative_prob=negative_prob)

        video = preprocess_video(video)
        return video

    return augment_video


def make_video_preprocess(height: int, width: int, base_channels: int):
    def preprocess(video: tf.Tensor, labels: tf.Tensor = None):
        video = tf.image.resize(video, (height, width))

        if base_channels == 3:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
            video *= rgb_weights
            video = tf.reduce_sum(video, axis=-1, keepdims=True)

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
