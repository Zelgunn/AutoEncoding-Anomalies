from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Layer, Dense, AveragePooling3D, UpSampling3D, Dropout
from tensorflow.python.keras.initializers import VarianceScaling
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlockND


# region Autoencoder
def make_residual_encoder(input_shape: Tuple[int, ...],
                          filters: List[int],
                          kernel_size: int,
                          strides: Union[List[Tuple[int, ...]], List[List[int]], List[int]],
                          code_size: int,
                          code_activation: Union[str, Layer],
                          model_depth: int,
                          seed=None,
                          name="ResidualEncoder"
                          ) -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    kwargs = {
        "rank": rank,
        "input_shape": input_shape,
        "activation": "relu",
        "kernel_size": kernel_size,
        "model_depth": model_depth,
        "seed": seed,
    }
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], basic_block_count=1, **kwargs)
        layers.append(layer)

        layer = AveragePooling3D(pool_size=strides[i])
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

    kwargs["kernel_size"] = 1
    kwargs["activation"] = code_activation
    # kwargs["kernel_initializer"] = VarianceScaling(scale=1.0, seed=seed)
    # kwargs.pop("model_depth")
    # kwargs.pop("seed")
    # layers.append(Conv(filters=code_size, strides=1, padding="same", **kwargs))
    layers.append(ResBlockND(filters=code_size, basic_block_count=2, **kwargs))

    return Sequential(layers=layers, name=name)


def make_residual_decoder(input_shape: Tuple[int, int, int, int],
                          filters: List[int],
                          kernel_size: int,
                          strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                          channels: int,
                          output_activation: Union[str, Layer],
                          model_depth: int,
                          seed=None,
                          name="ResidualDecoder"
                          ) -> Sequential:
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    kwargs = {
        "rank": rank,
        "activation": "relu",
        "kernel_size": kernel_size,
        "model_depth": model_depth,
        "seed": seed,
    }

    intermediate_shape = input_shape
    layers = []
    for i in range(len(filters)):
        layer = UpSampling3D(size=strides[i], input_shape=intermediate_shape)
        layers.append(layer)

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

        layer = ResBlockND(filters=filters[i], basic_block_count=1, **kwargs)
        layers.append(layer)

    kwargs["kernel_size"] = 1
    kwargs["activation"] = output_activation
    # kwargs["kernel_initializer"] = VarianceScaling(scale=0.5, seed=seed)
    # kwargs.pop("model_depth")
    # kwargs.pop("seed")
    # layers.append(Conv(filters=channels, strides=1, padding="same", use_bias=False, **kwargs))
    layers.append(ResBlockND(filters=channels, basic_block_count=2, **kwargs))

    return Sequential(layers=layers, name=name)


def make_discriminator(input_shape: Tuple[int, int, int, int],
                       filters: List[int],
                       kernel_size: int,
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       intermediate_size: int,
                       intermediate_activation: str,
                       include_intermediate_output: bool,
                       seed=None,
                       name="Discriminator"
                       ):
    # region Core : [ResBlock(), ResBlock(), ..., Dense()]
    validate_filters_and_strides(filters, strides)

    rank = len(input_shape) - 1
    kwargs = {
        "input_shape": input_shape,
        "rank": rank,
        "activation": "relu",
        "kernel_size": kernel_size,
        "model_depth": len(filters),
        "seed": seed,
    }

    dropout_rate = 0.0
    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], basic_block_count=1, **kwargs)
        layers.append(layer)

        layer = AveragePooling3D(pool_size=strides[i])
        layers.append(layer)

        if i == 0:
            kwargs.pop("input_shape")

        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

        layer = Dropout(rate=dropout_rate)
        layers.append(layer)

    dense_initializer = VarianceScaling(seed=seed)

    # layers.append(Flatten())
    layers.append(Dense(units=intermediate_size, activation=intermediate_activation,
                        kernel_initializer=dense_initializer))

    core_model = Sequential(layers, name="{}Core".format(name))
    # endregion

    input_layer = core_model.input
    intermediate_output = core_model.output
    final_output = Dense(units=1, activation="linear", kernel_initializer=dense_initializer)(intermediate_output)

    outputs = [final_output, intermediate_output] if include_intermediate_output else [final_output]

    return Model(inputs=[input_layer], outputs=outputs, name=name)


def validate_filters_and_strides(filters: List[int],
                                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]]
                                 ):
    if len(filters) != len(strides):
        raise ValueError("`filters` and `strides` must have the same length. "
                         "Found {} and {}".format(len(filters), len(strides)))

# endregion
