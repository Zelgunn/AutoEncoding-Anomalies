import tensorflow as tf
from tensorflow.python.keras import activations, Model
from tensorflow.python.keras.layers import Layer, Dense, InputSpec
from typing import List, Optional

from transformers.utils import get_look_ahead_mask
from transformers.core import MultiHeadAttention, AddAndNorm, PointWiseFeedForward


class SAAM(Model):
    def __init__(self,
                 layer_count: int,
                 head_count: int,
                 head_size: int,
                 intermediate_size: int,
                 output_size: int,
                 output_activation="linear",
                 **kwargs):
        super(SAAM, self).__init__(**kwargs)

        self.layer_count = layer_count
        self.head_count = head_count
        self.head_size = head_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.output_activation = activations.get(output_activation)

        self.projection_layer: Optional[Dense] = None
        self.attention_layers: List[MultiHeadAttention] = [SAAMBlock(head_count=head_count,
                                                                     head_size=head_size,
                                                                     intermediate_size=intermediate_size)
                                                           for _ in range(layer_count)]

        self.output_layer = Dense(units=output_size, use_bias=False, activation=self.output_activation)

    def call(self, inputs, training=None, mask=None):
        # region Flatten
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        input_size = inputs_shape[-1]
        inputs = tf.reshape(inputs, [batch_size, -1, input_size])
        # endregion

        # region Projection
        predicted_outputs = inputs
        if inputs.shape[-1] != self.intermediate_size:
            if self.projection_layer is None:
                self.projection_layer = Dense(units=self.intermediate_size, use_bias=False,
                                              name="projection_to_intermediate_size")
            predicted_outputs = self.projection_layer(inputs)
        # endregion

        # region Mask
        output_length = predicted_outputs.shape[1]
        if output_length is None:
            output_length = tf.shape(predicted_outputs)[1]
        look_ahead_mask = get_look_ahead_mask(output_length)
        look_ahead_mask = tf.expand_dims(look_ahead_mask, axis=0)
        # endregion

        # region Attention
        outputs = predicted_outputs
        for attention_layer in self.attention_layers:
            outputs = attention_layer(outputs, look_ahead_mask=look_ahead_mask)
        # endregion

        outputs = self.output_layer(outputs)

        # region Unflatten
        outputs_shape = tf.concat([inputs_shape[:-1], [self.output_size]], axis=0)
        outputs = tf.reshape(outputs, outputs_shape)
        # endregion

        return outputs

    def get_config(self):
        config = {
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "head_size": self.head_size,
            "intermediate_size": self.intermediate_size,
            "output_size": self.output_size,
            "output_activation": activations.serialize(self.output_activation),
        }
        return config


class SAAMBlock(Layer):
    def __init__(self,
                 head_count: int,
                 head_size: int,
                 intermediate_size: int,
                 **kwargs):
        super(SAAMBlock, self).__init__(**kwargs)

        self.head_count = head_count
        self.head_size = head_size
        self.intermediate_size = intermediate_size
        self.output_size: Optional[int] = None

        self.attention_layer: Optional[MultiHeadAttention] = None
        self.add_and_norm_attention = AddAndNorm()

        self.feed_forward: Optional[PointWiseFeedForward] = None
        self.add_and_norm_output = AddAndNorm()

    def build(self, input_shape):
        self.output_size = input_shape[-1]
        if self.output_size is None:
            raise ValueError("The last dimension of the inputs to `SAAMBlock` should be defined. Found `None`.")

        self.attention_layer = MultiHeadAttention(head_count=self.head_count,
                                                  keys_size=self.head_size,
                                                  values_size=self.head_size,
                                                  output_size=self.output_size)

        self.feed_forward = PointWiseFeedForward(intermediate_size=self.intermediate_size,
                                                 output_size=self.output_size)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.output_size})
        super(SAAMBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        look_ahead_mask = kwargs["look_ahead_mask"] if "look_ahead_mask" in kwargs else None

        attention_inputs = [inputs, inputs, inputs]
        attention_outputs, _ = self.attention_layer(attention_inputs, mask=look_ahead_mask)
        attention_outputs = self.add_and_norm_attention([inputs, attention_outputs])

        outputs = self.feed_forward(attention_outputs)
        outputs = self.add_and_norm_output([attention_outputs, outputs])

        return outputs

    def get_config(self):
        config = {
            "head_count": self.head_count,
            "head_size": self.head_size,
            "intermediate_size": self.intermediate_size,
            "output_size": self.output_size,
        }
        return config
