from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.init_ops import VarianceScaling
from typing import List, Tuple, Union

from CustomKerasLayers import ResBlockND, ResSASABlock


def make_encoder(input_shape: Tuple[int, ...],
                 mode: str,
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 strides: Union[List[Tuple[int, ...]], List[List[int]], List[int]],
                 code_size: int,
                 code_activation: Union[str, Layer],
                 basic_block_count=1,
                 seed=None,
                 name="Encoder"
                 ) -> Sequential:
    layers = get_encoder_layers(rank=len(input_shape) - 1,
                                mode=mode,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                code_size=code_size,
                                code_activation=code_activation,
                                basic_block_count=basic_block_count,
                                seed=seed)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_decoder(input_shape: Tuple[int, ...],
                 mode: str,
                 filters: List[int],
                 kernel_size: Union[int, List[int]],
                 strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                 channels: int,
                 output_activation: Union[str, Layer],
                 basic_block_count=1,
                 seed=None,
                 name="Decoder"
                 ) -> Sequential:
    layers = get_decoder_layers(rank=len(input_shape) - 1,
                                mode=mode,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                channels=channels,
                                output_activation=output_activation,
                                basic_block_count=basic_block_count,
                                seed=seed)

    return to_sequential(layers=layers, input_shape=input_shape, name=name)


def make_discriminator(input_shape: Tuple[int, ...],
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       intermediate_size: int,
                       intermediate_activation: str,
                       include_intermediate_output: bool,
                       basic_block_count=1,
                       seed=None,
                       name="Discriminator"
                       ):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(filters)

    rank = len(input_shape) - 1
    kwargs = {
        "input_shape": input_shape,
        "rank": rank,
        "basic_block_count": basic_block_count,
        "seed": seed,
    }

    intermediate_shape = input_shape

    layers = []
    for i in range(len(filters)):
        layer = ResBlockND(filters=filters[i], kernel_size=kernel_size[i], **kwargs)
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
                       mode: str,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       code_size: int,
                       code_activation: Union[str, Layer],
                       seed: int,
                       **kwargs
                       ) -> List[Layer]:
    layer_count = len(filters)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * layer_count
    shared_params = {"rank": rank, "transposed": False, "mode": mode, "use_bias": True, "seed": seed, **kwargs}

    layers = []
    for i in range(layer_count):
        is_first_layer = (i == 0)
        layer_kernel_size = kernel_size[i] + 4 if is_first_layer else kernel_size[i]
        layer_activation = "relu" if mode == "conv" else "linear"
        layer = get_layer(filters=filters[i], kernel_size=layer_kernel_size, strides=strides[i],
                          activation=layer_activation, **shared_params)
        layers += layer

    latent_code_layer = get_layer(filters=code_size, kernel_size=1, strides=1,
                                  activation=code_activation, name="LatentCodeLayer", **shared_params)
    layers += latent_code_layer
    return layers


def get_decoder_layers(rank: int,
                       mode: str,
                       filters: List[int],
                       kernel_size: Union[int, List[int]],
                       strides: Union[List[Tuple[int, int, int]], List[List[int]], List[int]],
                       channels: int,
                       output_activation: Union[str, Layer],
                       seed: int,
                       **kwargs,
                       ) -> List[Layer]:
    layer_count = len(filters)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * layer_count
    shared_params = {"rank": rank, "transposed": True, "mode": mode, "use_bias": True, "seed": seed, **kwargs}

    layers = []
    for i in range(layer_count):
        is_last_layer = (i == (layer_count - 1))
        layer_kernel_size = kernel_size[i] + 4 if is_last_layer else kernel_size[i]
        layer_activation = "relu" if mode == "conv" else "linear"
        layer = get_layer(filters=filters[i], kernel_size=layer_kernel_size, strides=strides[i],
                          activation=layer_activation, **shared_params)
        layers += layer

    if "basic_block_count" in shared_params:
        shared_params["basic_block_count"] = 1

    output_layer = get_layer(filters=channels, kernel_size=1, strides=1,
                             activation=output_activation, name="OutputLayer", **shared_params)
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
              seed: int,
              name=None,
              **kwargs
              ) -> List[Layer]:
    if rank not in [1, 2, 3]:
        raise ValueError("`rank` must be in [1, 2, 3]. Got {}".format(rank))

    padding = dict_get(kwargs, "padding", default="same")
    if mode == "conv":
        kernel_initializer = dict_get(kwargs, "kernel_initializer", default=VarianceScaling(seed=seed))
        main_layer = Conv(rank=rank, filters=filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                          strides=1, padding=padding, activation=activation, use_bias=use_bias,
                          name=name)
    elif mode == "residual":
        basic_block_count = dict_get(kwargs, "basic_block_count", default=1)
        main_layer = ResBlockND(rank=rank, filters=filters, kernel_size=kernel_size,
                                basic_block_count=basic_block_count, strides=1, padding=padding,
                                activation=activation, seed=seed, name=name)
    elif mode == "residual_self_attention":
        head_count = dict_get(kwargs, "head_count", default=8)
        head_size = filters // head_count
        basic_block_count = dict_get(kwargs, "basic_block_count", default=1)
        main_layer = ResSASABlock(rank=rank, head_size=head_size, head_count=head_count, kernel_size=kernel_size,
                                  basic_block_count=basic_block_count, strides=1,
                                  activation=activation, seed=seed, name=name)
    else:
        raise ValueError("`mode` must be in ['conv', 'residual', 'residual_self_attention']. Got {}".format(mode))

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
