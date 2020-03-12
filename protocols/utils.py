from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow.python.keras.initializers import VarianceScaling
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlockND


def make_encoder(input_shape: Tuple[int, ...],
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 strides: Union[List[Tuple[int, ...]], List[List[int]], List[int]],
                 code_size: int,
                 code_activation: Union[str, Layer],
                 model_depth: int,
                 seed=None,
                 name="ResidualEncoder"
                 ) -> Sequential:
    layers = get_encoder_layers(rank=len(input_shape) - 1,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                code_size=code_size,
                                code_activation=code_activation,
                                model_depth=model_depth,
                                seed=seed)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_decoder(input_shape: Tuple[int, ...],
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                 channels: int,
                 output_activation: Union[str, Layer],
                 model_depth: int,
                 seed=None,
                 name="ResidualDecoder"
                 ) -> Sequential:
    layers = get_decoder_layers(rank=len(input_shape) - 1,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                channels=channels,
                                output_activation=output_activation,
                                model_depth=model_depth,
                                seed=seed)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_discriminator(input_shape: Tuple[int, ...],
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       intermediate_size: int,
                       intermediate_activation: str,
                       include_intermediate_output: bool,
                       seed=None,
                       name="Discriminator"
                       ):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(filters)

    rank = len(input_shape) - 1
    kwargs = {
        "input_shape": input_shape,
        "rank": rank,
        "activation": "relu",
        "model_depth": len(filters),
        "seed": seed,
    }

    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], basic_block_count=1, kernel_size=kernel_size[i], **kwargs)
        layers.append(layer)

        layer = average_pooling(rank=rank, pool_size=strides[i])
        layers.append(layer)
        intermediate_shape = layer.compute_output_shape((None, *intermediate_shape))[1:]

        if i == 0:
            kwargs.pop("input_shape")

    dense_initializer = VarianceScaling(seed=seed)

    layers.append(Dense(units=intermediate_size, activation=intermediate_activation,
                        kernel_initializer=dense_initializer))

    core_model = Sequential(layers, name="{}Core".format(name))

    input_layer = core_model.input
    intermediate_output = core_model.output
    final_output = Dense(units=1, activation="linear", kernel_initializer=dense_initializer)(intermediate_output)

    outputs = [final_output, intermediate_output] if include_intermediate_output else [final_output]

    return Model(inputs=[input_layer], outputs=outputs, name=name)


def get_encoder_layers(rank: int,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, ...]], List[List[int]], List[int]],
                       code_size: int,
                       code_activation: Union[str, Layer],
                       model_depth: int,
                       seed=None) -> List[Layer]:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(filters)

    kwargs = {
        "rank": rank,
        "activation": "relu",
        "model_depth": model_depth,
        "seed": seed,
    }

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], basic_block_count=1, kernel_size=kernel_size[i], **kwargs)
        layers.append(layer)

        layer = average_pooling(rank=rank, pool_size=strides[i])
        layers.append(layer)

    kwargs["activation"] = code_activation
    layers.append(ResBlockND(filters=code_size, basic_block_count=2, kernel_size=1, **kwargs))
    return layers


def get_decoder_layers(rank: int,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       channels: int,
                       output_activation: Union[str, Layer],
                       model_depth: int,
                       seed=None,
                       ) -> List[Layer]:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(filters)

    kwargs = {
        "rank": rank,
        "activation": "relu",
        "model_depth": model_depth,
        "seed": seed,
    }

    layers = []
    for i in range(len(filters)):
        layer = upsampling(rank=rank, size=strides[i])
        layers.append(layer)

        layer = ResBlockND(filters=filters[i], basic_block_count=1, kernel_size=kernel_size[i], **kwargs)
        layers.append(layer)

    kwargs["activation"] = output_activation
    layers.append(ResBlockND(filters=channels, basic_block_count=2, kernel_size=1, **kwargs))
    return layers


def to_sequential(layers: List[Layer], input_shape: Tuple[int, ...], name: str) -> Sequential:
    # noinspection PyProtectedMember
    layers[0]._batch_input_shape = (None, *input_shape)
    return Sequential(layers=layers, name=name)


def upsampling(rank: int,
               size: Tuple[int, ...]
               ) -> Union[UpSampling1D, UpSampling2D, UpSampling3D]:
    upsampling_classes = {1: UpSampling1D, 2: UpSampling2D, 3: UpSampling3D}
    upsampling_class = upsampling_classes[rank]
    return upsampling_class(size=size)


def average_pooling(rank: int,
                    pool_size: Tuple[int, ...]
                    ) -> Union[AveragePooling1D, AveragePooling2D, AveragePooling3D]:
    average_pooling_classes = {1: AveragePooling1D, 2: AveragePooling2D, 3: AveragePooling3D}
    average_pooling_class = average_pooling_classes[rank]
    return average_pooling_class(pool_size=pool_size)
