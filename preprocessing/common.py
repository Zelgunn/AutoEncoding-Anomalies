import tensorflow as tf


def dropout_noise(inputs, max_rate, noise_shape_prob):
    """
    :param inputs: A floating point `Tensor`.
    :param max_rate: A floating point scalar `Tensor`. The maximum probability that each element is dropped.
    :param noise_shape_prob: A 1-D `Tensor` of type `float32`, representing the probability of dropping each
        dimension completely.
    :return: A `Tensor` of the same shape and type as `inputs.
    """
    noise_shape = tf.random.uniform(shape=[len(inputs.shape)], minval=0.0, maxval=1.0, dtype=tf.float32)
    noise_shape = noise_shape < noise_shape_prob
    noise_shape = tf.cast(noise_shape, tf.int32)
    noise_shape = tf.shape(inputs) * (1 - noise_shape) + noise_shape

    rate = tf.random.uniform(shape=[], minval=0.0, maxval=max_rate, dtype=tf.float32)
    random_tensor = tf.random.uniform(shape=noise_shape, minval=0.0, maxval=1.0, dtype=tf.float32)
    keep_mask = random_tensor >= rate

    outputs = inputs * tf.cast(keep_mask, inputs.dtype) / (1.0 - rate)
    return outputs
