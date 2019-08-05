import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


def squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.square(y_true - y_pred)


def absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.abs(y_true - y_pred)


def negative_ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes one minus SSIM between `y_true` and `y_pred`.

    Values of `y_true` and `y_pred` are supposed to be between 0.0 and 1.0.

    :param y_true: A 3D+ tensor with dtype `tf.float32`.
    :param y_pred: A 3D+ tensor with dtype `tf.float32`.
    :return: A tensor with the same shape as `y_true` and `y_pred`, minus the last 3 dimensions.
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)


class RawPredictionsLayer(Layer):
    """
    Layer class for making raw predictions for anomaly detection.

    Arguments:
        metric: Either a string ("mse", "mae", "ssim") or a function that takes two arguments
            (`y_true` and `y_pred`) and returns a single tensor containing the value of the custom metric.
        output_length: If specified, the inputs and outputs of this layer will be split at the specified length.
        kwargs: Optional parameter for the base class (Layer).
    """
    def __init__(self,
                 metric="mse",
                 output_length: int = None,
                 **kwargs):

        super(RawPredictionsLayer, self).__init__(trainable=False, **kwargs)
        if isinstance(metric, str):
            if metric is "mse":
                metric = squared_error
            elif metric is "mae":
                metric = absolute_error
            elif metric is "ssim":
                metric = negative_ssim
            else:
                raise ValueError("Unknown metric : {}".format(metric))

        self.metric = metric
        self.output_length = output_length

    def call(self, inputs, **kwargs):
        pred_output, true_output = inputs

        if self.output_length is None:
            length = tf.shape(true_output)[1]
        else:
            length = self.output_length
            true_output = true_output[:, :length]

        pred_output = pred_output[:, :length]

        error = self.metric(true_output, pred_output)
        reduction_axis = tuple(range(2, len(error.shape)))
        if len(reduction_axis) != 0:
            error = tf.reduce_mean(error, axis=reduction_axis)
        return error

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(shape=input_signature.shape[:2], dtype=tf.float32)
