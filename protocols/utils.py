from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LeakyReLU, Conv3D, Layer
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlock3D, ResBlock3DTranspose


def make_residual_encoder(filters: List[int],
                          strides: List[Tuple[int, int, int]],
                          code_size: int,
                          code_activation: Union[str, Layer],
                          name="ResidualEncoder") -> Sequential:
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length")

    leaky_relu = LeakyReLU(alpha=1e-2)

    layers = []
    for i in range(len(filters)):
        layer = ResBlock3D(filters=filters[i], strides=strides[i], kernel_size=3, activation=leaky_relu)
        layers.append(layer)
    layers.append(ResBlock3D(filters=code_size, strides=1, kernel_size=3, activation=code_activation))

    return Sequential(layers=layers, name=name)


def make_residual_decoder(filters: List[int],
                          strides: List[Tuple[int, int, int]],
                          channels: int,
                          output_activation: Union[str, Layer],
                          name="ResidualDecoder") -> Sequential:
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length")

    leaky_relu = LeakyReLU(alpha=1e-2)

    layers = []
    for i in range(len(filters)):
        layer = ResBlock3DTranspose(filters=filters[i], strides=strides[i], kernel_size=3, activation=leaky_relu)
        layers.append(layer)
    layers.append(Conv3D(filters=channels, strides=1, kernel_size=1, padding="same", activation=output_activation,
                         use_bias=False))

    return Sequential(layers=layers, name=name)
