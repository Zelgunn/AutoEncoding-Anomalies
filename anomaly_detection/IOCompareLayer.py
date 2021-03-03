import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


def squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.square(y_true - y_pred)


def absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.abs(y_true - y_pred)


def log_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mae = absolute_error(y_true, y_pred)
    return tf.math.log(mae + tf.constant(1.0))


def clipped_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mae = absolute_error(y_true, y_pred)
    return tf.clip_by_value(mae, 0.0, 1.0)


def clipped_mae_32(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mae = absolute_error(y_true, y_pred)
    return tf.clip_by_value(mae, 0.0, 32.0)


def negative_ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes one minus SSIM between `y_true` and `y_pred`.

    Values of `y_true` and `y_pred` are supposed to be between 0.0 and 1.0.

    :param y_true: A 3D+ tensor with dtype `tf.float16` or `tf.float32`.
    :param y_pred: A 3D+ tensor with dtype `tf.float16` or `tf.float32`.
    :return: A tensor with the same shape as `y_true` and `y_pred`, minus the last 3 dimensions, and the same
        dtype as inputs.
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)


def negative_psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes negative PSNR between `y_true` and `y_pred`.

    Values of `y_true` and `y_pred` are supposed to be between 0.0 and 1.0.

    :param y_true: A 3D+ tensor with dtype `tf.float16` or `tf.float32`.
    :param y_pred: A 3D+ tensor with dtype `tf.float16` or `tf.float32`.
    :return: A tensor with the same shape as `y_true` and `y_pred`, minus the last 3 dimensions, and the same
        dtype as inputs.
    """
    return - tf.image.psnr(y_true, y_pred, max_val=1.0)


known_metrics = {
    "mse": squared_error,
    "mae": absolute_error,
    # "ssim": negative_ssim,
    # "psnr": negative_psnr,
    "log_mae": log_absolute_error,
    "clipped_mae": clipped_mae,
    "clipped_mae_32": clipped_mae_32,
}


class IOCompareLayer(Layer):
    """
    Layer class for making raw predictions for anomaly detection, using a comparison metric.

    Arguments:
        metric: Either a string ("mse", "mae", "ssim", "psnr") or a function that takes two arguments
            (`y_true` and `y_pred`) and returns a single tensor containing the value of the custom metric.
        output_length: If specified, the inputs and outputs of this layer will be split at the specified length.
        kwargs: Optional parameter for the base class (Layer).
    """

    def __init__(self,
                 metric="mse",
                 **kwargs):
        super(IOCompareLayer, self).__init__(trainable=False, **kwargs)

        if isinstance(metric, str):
            if metric in known_metrics:
                metric = known_metrics[metric]
            else:
                raise ValueError("Unknown metric : {}. Known metrics are : {}"
                                 .format(metric, list(known_metrics.keys())))

        self.metric = metric

    def call(self, inputs, **kwargs):
        pred_output, true_output = inputs

        error = self.metric(true_output, pred_output)
        reduction_axis = tuple(range(2, len(error.shape)))
        if len(reduction_axis) != 0:
            error = tf.reduce_mean(error, axis=reduction_axis)
        return error

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(shape=input_signature.shape[:2], dtype=tf.float32)
