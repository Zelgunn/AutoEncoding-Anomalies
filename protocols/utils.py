from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LeakyReLU, Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils import conv_utils
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlockND, ResBlockNDTranspose


def make_residual_encoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          code_size: int,
                          code_activation: Union[str, Layer],
                          use_batch_norm: bool,
                          name="ResidualEncoder") -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "activation": leaky_relu,
        "use_batch_norm": use_batch_norm,
        "rank": rank,
    }
    intermediate_shape = input_shape
    kernel_size = get_kernel_size(3, intermediate_shape)

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], strides=strides[i], kernel_size=kernel_size, **kwargs)
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]
        kernel_size = get_kernel_size(3, intermediate_shape)

    kwargs["activation"] = code_activation
    layers.append(ResBlockND(filters=code_size, strides=1, kernel_size=kernel_size, **kwargs))

    return Sequential(layers=layers, name=name)


def make_residual_decoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          channels: int,
                          output_activation: Union[str, Layer],
                          use_batch_norm: bool,
                          name="ResidualDecoder") -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    leaky_relu = LeakyReLU(alpha=1e-2)
    kwargs = {
        "input_shape": input_shape,
        "activation": leaky_relu,
        "use_batch_norm": use_batch_norm,
        "rank": rank,
    }
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        kernel_size = get_kernel_size(3, intermediate_shape)
        layer = ResBlockNDTranspose(filters=filters[i], strides=strides[i], kernel_size=kernel_size, **kwargs)
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

    kwargs.pop("use_batch_norm")
    kwargs["kernel_size"] = 1
    kwargs["activation"] = output_activation
    layers.append(Conv(filters=channels, strides=1, padding="same", use_bias=False, **kwargs))

    return Sequential(layers=layers, name=name)


def validate_filters_and_strides(filters: List[int],
                                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]]
                                 ):
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length. "
                         "Found {} and {}".format(len(filters), len(strides)))


def get_kernel_size(max_kernel_size, input_shape):
    rank = len(input_shape) - 1
    max_kernel_size = conv_utils.normalize_tuple(max_kernel_size, rank, "kernel_size")
    kernel_size = tuple([min(max_kernel_size[i], input_shape[i]) for i in range(rank)])
    return kernel_size
