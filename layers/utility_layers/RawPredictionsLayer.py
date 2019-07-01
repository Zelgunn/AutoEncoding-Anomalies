import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from typing import Union, List, Tuple


class RawPredictionsLayer(Layer):
    def __init__(self,
                 reduction_axis: Union[List[int], Tuple[int, ...], int],
                 output_length: int = None,
                 **kwargs):
        super(RawPredictionsLayer, self).__init__(trainable=False,
                                                  **kwargs)
        self.reduction_axis = reduction_axis
        self.output_length = output_length

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        pred_output, true_output = inputs

        if self.output_length is None:
            length = tf.shape(true_output)[1]
        else:
            length = self.output_length
            true_output = true_output[:, :length]

        pred_output = pred_output[:, :length]

        error = tf.square(pred_output - true_output)
        error = tf.reduce_mean(error, axis=self.reduction_axis)
        return error
