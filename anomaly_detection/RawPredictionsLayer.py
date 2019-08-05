import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


def squared_error(y_true, y_pred):
    return tf.square(y_true - y_pred)


def absolute_error(y_true, y_pred):
    return tf.abs(y_true - y_pred)


class RawPredictionsLayer(Layer):
    def __init__(self,
                 metric="mse",
                 output_length: int = None,
                 **kwargs):
        super(RawPredictionsLayer, self).__init__(trainable=False,
                                                  **kwargs)
        if metric is "mse":
            metric = squared_error
        elif metric is "mae":
            metric = absolute_error

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
        error = tf.reduce_mean(error, axis=reduction_axis)
        return error

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return tf.TensorSpec(shape=input_signature.shape[:2], dtype=tf.float32)
