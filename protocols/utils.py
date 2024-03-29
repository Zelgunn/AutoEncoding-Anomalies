from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Layer, Dense, Flatten
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow.python.ops.init_ops import VarianceScaling
from typing import List, Tuple, Union, Type

from CustomKerasLayers import ResBlockND


def make_encoder(input_shape: Tuple[int, ...],
                 mode: str,
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 strides: Union[List[Tuple[int, ...]], List[List[int]], List[int]],
                 code_size: int,
                 code_activation: Union[str, Layer],
                 use_code_bias: bool,
                 basic_block_count=1,
                 flatten_code=False,
                 name="Encoder"
                 ) -> Sequential:
    layers = get_encoder_layers(rank=len(input_shape) - 1,
                                mode=mode,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                code_size=code_size,
                                code_activation=code_activation,
                                use_code_bias=use_code_bias,
                                basic_block_count=basic_block_count,
                                flatten_code=flatten_code,
                                name=name)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_decoder(input_shape: Tuple[int, ...],
                 mode: str,
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 stem_kernel_size: int,
                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                 channels: int,
                 output_activation: Union[str, Layer],
                 basic_block_count=1,
                 name="Decoder"
                 ) -> Sequential:
    layers = get_decoder_layers(rank=len(input_shape) - 1,
                                mode=mode,
                                filters=filters,
                                kernel_size=kernel_size,
                                stem_kernel_size=stem_kernel_size,
                                strides=strides,
                                channels=channels,
                                output_activation=output_activation,
                                basic_block_count=basic_block_count,
                                name=name)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_discriminator(input_shape: Tuple[int, ...],
                       mode: str,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       intermediate_size: int,
                       intermediate_activation: str,
                       include_intermediate_output: bool,
                       basic_block_count=1,
                       name="Discriminator"
                       ):
    core_model = make_encoder(input_shape=input_shape, mode=mode, filters=filters, kernel_size=kernel_size,
                              strides=strides, code_size=intermediate_size, code_activation=intermediate_activation,
                              use_code_bias=False, basic_block_count=basic_block_count, flatten_code=False, name=name)

    input_layer = core_model.input
    intermediate_output = core_model.output

    flatten_layer = Flatten(name="{}_Flatten".format(name))
    final_layer = Dense(units=1, activation="linear", kernel_initializer="he_normal", use_bias=False,
                        name="{}_Output".format(name))
    final_output = final_layer(flatten_layer(intermediate_output))

    outputs = [final_output, intermediate_output] if include_intermediate_output else [final_output]

    return Model(inputs=[input_layer], outputs=outputs, name=name)


def get_encoder_layers(rank: int,
                       mode: str,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       code_size: int,
                       code_activation: Union[str, Layer],
                       use_code_bias: bool,
                       flatten_code: bool,
                       name: str,
                       **kwargs
                       ) -> List[Layer]:
    layers = []

    layer_count = len(filters)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * layer_count
    shared_params = {"rank": rank, "transposed": False, "mode": mode, **kwargs}

    layer_activation = "linear" if mode == "residual" else "relu"
    for i in range(layer_count):
        layer = get_layer(filters=filters[i], kernel_size=kernel_size[i], strides=strides[i], use_bias=True,
                          activation=layer_activation, name="{}_{}".format(name, i), **shared_params)
        layers += layer

    latent_code_layer = get_layer(filters=code_size, kernel_size=1, strides=1, use_bias=use_code_bias,
                                  activation=code_activation, name="{}_LatentCodeLayer".format(name), **shared_params)
    layers += latent_code_layer

    if flatten_code:
        layers += [Flatten()]

    return layers


def get_decoder_layers(rank: int,
                       mode: str,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       stem_kernel_size: int,
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       channels: int,
                       output_activation: Union[str, Layer],
                       name: str,
                       **kwargs,
                       ) -> List[Layer]:
    layers = []

    layer_count = len(filters)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * layer_count
    shared_params = {"rank": rank, "transposed": True, "mode": mode, "use_bias": True, **kwargs}

    layer_activation = "linear" if mode == "residual" else "relu"
    for i in range(layer_count):
        layers += get_layer(filters=filters[i], kernel_size=kernel_size[i], strides=strides[i],
                            activation=layer_activation, name="{}_{}".format(name, i), **shared_params)

    output_layer = get_layer(filters=channels, kernel_size=stem_kernel_size, strides=1,
                             activation=output_activation, name="{}_OutputLayer".format(name), **shared_params)
    layers += output_layer
    return layers


def get_layer(rank: int,
              transposed: bool,
              mode: str,
              filters: int,
              kernel_size: int,
              strides: Union[Tuple[int, int, int], List[int], int],
              use_bias: bool,
              activation: Union[str, Layer],
              name=None,
              **kwargs
              ) -> List[Layer]:
    if rank not in [1, 2, 3] and mode in ["conv", "residual"]:
        raise ValueError("`rank` must be in [1, 2, 3]. Got {}".format(rank))

    padding = dict_get(kwargs, "padding", default="same")
    if mode == "conv":
        kernel_initializer = dict_get(kwargs, "kernel_initializer", default=VarianceScaling())
        conv_class = get_conv_class(rank)
        main_layer = conv_class(filters=filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                                strides=1, padding=padding, activation=activation, use_bias=use_bias,
                                name=name)
    elif mode == "dense":
        kernel_initializer = dict_get(kwargs, "kernel_initializer", default=VarianceScaling())
        main_layer = Dense(units=filters, activation=activation, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, name=name)
    elif mode == "residual":
        basic_block_count = dict_get(kwargs, "basic_block_count", default=1)
        main_layer = ResBlockND(rank=rank, filters=filters, kernel_size=kernel_size,
                                basic_block_count=basic_block_count, strides=1, padding=padding,
                                activation=activation, name=name)
    else:
        raise ValueError("`mode` must be in ['conv', 'dense', 'residual']. Got {}".format(mode))

    if isinstance(strides, int):
        no_stride = strides == 1
    else:
        no_stride = all(x == 1 for x in strides)

    if no_stride:
        layers = [main_layer]
    elif transposed:
        layers = [upsampling(rank=rank, size=strides), main_layer]
    else:
        layers = [main_layer, average_pooling(rank=rank, pool_size=strides)]

    return layers


def to_sequential(layers: List[Layer], input_shape: Tuple[int, ...], name: str) -> Sequential:
    # noinspection PyProtectedMember
    layers[0]._batch_input_shape = (None, *input_shape)
    return Sequential(layers=layers, name=name)


def get_conv_class(rank: int) -> Union[Type[Conv1D], Type[Conv2D], Type[Conv3D]]:
    conv_classes = {1: Conv1D, 2: Conv2D, 3: Conv3D}
    return conv_classes[rank]


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


def dict_get(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        return default
