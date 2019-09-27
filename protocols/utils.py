from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LeakyReLU, Conv3D, Layer
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlock3D, ResBlock3DTranspose


def make_residual_encoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          code_size: int,
                          code_activation: Union[str, Layer],
                          name="ResidualEncoder") -> Sequential:
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length")

    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "kernel_size": 3,
        "activation": leaky_relu,
    }

    layers = []
    for i in range(len(filters)):
        layer = ResBlock3D(filters=filters[i], strides=strides[i], **kwargs)
        layers.append(layer)
        if i == 0:
            kwargs.pop("input_shape")
    layers.append(ResBlock3D(filters=code_size, strides=1, kernel_size=3, activation=code_activation))

    return Sequential(layers=layers, name=name)


def make_residual_decoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          channels: int,
                          output_activation: Union[str, Layer],
                          name="ResidualDecoder") -> Sequential:
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length")

    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "kernel_size": 3,
        "activation": leaky_relu,
    }

    layers = []
    for i in range(len(filters)):
        layer = ResBlock3DTranspose(filters=filters[i], strides=strides[i], **kwargs)
        layers.append(layer)
        if i == 0:
            kwargs.pop("input_shape")
    layers.append(Conv3D(filters=channels, strides=1, kernel_size=1, padding="same", activation=output_activation,
                         use_bias=False))

    return Sequential(layers=layers, name=name)
