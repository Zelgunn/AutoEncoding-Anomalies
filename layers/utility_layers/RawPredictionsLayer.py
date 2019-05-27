import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


class RawPredictionsLayer(Layer):
    def __init__(self, reduction_axis, **kwargs):
        super(RawPredictionsLayer, self).__init__(trainable=False,
                                                  **kwargs)
        self.reduction_axis = reduction_axis

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        pred_output, true_output = inputs

        length = tf.shape(true_output)[1]
        pred_output = pred_output[:, :length]

        error = tf.square(pred_output - true_output)
        error = tf.reduce_mean(error, axis=self.reduction_axis)
        return error
