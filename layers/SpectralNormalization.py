from keras.layers import Wrapper, InputSpec
from keras.layers import Dense, Conv2D, Deconv2D
from keras.initializers import RandomNormal
from typing import List
import keras.backend as K
import tensorflow as tf

from layers import ResBlock1D, ResBlock2D, ResBlock3D, ResBlock2DTranspose, ResBlock3DTranspose


class WeightsInfo(object):
    def __init__(self, name: str, weights: tf.Variable, u: tf.Variable, index: int = None):
        self.name = name
        self.weights = weights
        self.u = u
        self.index = index


KNOWN_WEIGHTS_NAMES = {Dense: "kernel",
                       Conv2D: "kernel",
                       Deconv2D: "kernel",
                       ResBlock1D: "kernels",
                       ResBlock2D: "kernels",
                       ResBlock3D: "kernels",
                       ResBlock2DTranspose: "kernels",
                       ResBlock3DTranspose: "kernels"}


class SpectralNormalization(Wrapper):
    def __init__(self, layer, norm_weight_names: str or List = None):
        super(SpectralNormalization, self).__init__(layer)
        if norm_weight_names is None:
            assert layer.__class__ in KNOWN_WEIGHTS_NAMES, "Class weights are unknown, you must provide weights names"
            norm_weight_names = KNOWN_WEIGHTS_NAMES[layer.__class__]
        self.norm_weights_names = norm_weight_names if isinstance(norm_weight_names, list) else [norm_weight_names]
        self.u_initializer = RandomNormal(mean=0.0, stddev=1.0)
        self.weights_infos = []
        self.layer_weights_updated = False

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = self.layer.input_spec

        for name in self.norm_weights_names:
            weights = getattr(self.layer, name)
            if isinstance(weights, list):
                for i in range(len(weights)):
                    weights_info = self.make_u_variable(name, weights[i], i)
                    self.weights_infos.append(weights_info)
            else:
                weights_info = self.make_u_variable(name, weights)
                self.weights_infos.append(weights_info)

        super(SpectralNormalization, self).build()

    def make_u_variable(self, name: str, weights: tf.Variable, index=None) -> WeightsInfo:
        dim = weights.shape.as_list()[-1]
        u = self.add_weight(name=name + "_spectral_norm",
                            shape=[1, dim],
                            initializer=self.u_initializer,
                            trainable=False)
        return WeightsInfo(name, weights, u, index)

    def call(self, inputs, **kwargs):
        assert "training" not in kwargs
        if not self.layer_weights_updated:
            self.update_layer_weights()
            self.layer_weights_updated = True
        return self.layer.call(inputs)

    def update_layer_weights(self):
        for weights_info in self.weights_infos:
            weights_normalized = SpectralNormalization.normalize_weights(weights_info)
            if weights_info.index is None:
                setattr(self.layer, weights_info.name, weights_normalized)
            else:
                weights_list = getattr(self.layer, weights_info.name)
                weights_list[weights_info.index] = weights_normalized

    @staticmethod
    def normalize_weights(weights_info: WeightsInfo):
        u_variable = weights_info.u
        weights = weights_info.weights
        weights_shape = weights.shape.as_list()
        reshaped_weights = K.reshape(weights, [-1, weights_shape[-1]])

        u, v = SpectralNormalization.power_iteration(reshaped_weights, u_variable)
        sigma = v @ reshaped_weights @ K.transpose(u)
        weights_normalized = reshaped_weights / sigma

        with tf.control_dependencies([u_variable.assign(u)]):
            weights_normalized = K.reshape(weights_normalized, weights_shape)

        return weights_normalized

    @staticmethod
    def power_iteration(weights: tf.Variable, u: tf.Variable):
        v = K.l2_normalize(u @ K.transpose(weights))
        u = K.l2_normalize(v @ weights)
        return u, v

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
