import tensorflow as tf

from data_processing.common import dropout_noise


def make_video_augmentation(length: int, height: int, width: int, channels: int,
                            seed,
                            dropout_noise_ratio=0.0,
                            negative_prob=0.0,
                            ):
    to_grayscale = channels == 3

    def augment_video(video: tf.Tensor) -> tf.Tensor:
        video = tf.image.random_crop(video, size=(length, height, width, channels), seed=seed)

        if dropout_noise_ratio > 0.0:
            video = video_dropout_noise(video, dropout_noise_ratio, spatial_prob=0.1, seed=seed)

        if negative_prob > 0.0:
            video = random_image_negative(video, negative_prob=negative_prob, seed=seed)

        if to_grayscale:
            video = convert_to_grayscale(video)

        video = tf.image.per_image_standardization(video)
        # video = normalize_tanh(video)
        return video

    return augment_video


def make_video_preprocess(height: int, width: int, to_grayscale: bool):
    def preprocess(video: tf.Tensor, labels: tf.Tensor = None):
        # video = tf.image.resize(video, (height, width))

        if to_grayscale:
            video = convert_to_grayscale(video)

        video = tf.image.per_image_standardization(video)
        # video = normalize_tanh(video)

        if labels is None:
            return video
        else:
            return video, labels

    return preprocess


def video_random_cropping(video: tf.Tensor, crop_ratio: float, output_length: int, seed=None):
    crop_ratio = tf.random.uniform(shape=(), minval=1.0 - crop_ratio, maxval=1.0, seed=seed)
    original_shape = tf.cast(tf.shape(video), tf.float32)
    original_height, original_width = original_shape[1], original_shape[2]

    crop_size = [output_length, crop_ratio * original_height, crop_ratio * original_width, 1]

    video = tf.image.random_crop(video, crop_size, seed=seed)
    return video


def video_dropout_noise(video, max_rate, spatial_prob, seed=None):
    drop_width = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed) > 0.5
    drop_width = tf.cast(drop_width, tf.float32)

    noise_shape_prob = [spatial_prob * (1.0 - drop_width), spatial_prob * drop_width, 1.0]
    noise_shape_prob = [0.0] * (len(video.shape) - 3) + noise_shape_prob

    video = dropout_noise(video, max_rate, noise_shape_prob, seed=seed)
    return video


def random_image_negative(image, negative_prob=0.5, seed=None):
    negative = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed) < negative_prob
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


@tf.function
def normalize_sigmoid(x: tf.Tensor) -> tf.Tensor:
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


@tf.function
def normalize_tanh(x: tf.Tensor) -> tf.Tensor:
    x = normalize_sigmoid(x)
    x = tf.constant(2.0) * x - tf.constant(1.0)
    return x
