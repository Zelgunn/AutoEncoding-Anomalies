from tensorflow.python.keras.layers import Conv2DTranspose


def make_deconv_layers_with_target(input_shape,
                                   target_shape,
                                   depth: int,
                                   filters,
                                   # kernels_estimation="closest",
                                   strides,
                                   activations=None,
                                   tile_rank_over_depth=True
                                   ):
    rank = len(input_shape)
    parameters_length = check_parameters_lengths(strides, filters, activations)
    assert (parameters_length is None) or (parameters_length == depth) or (parameters_length == rank), \
        "Length of parameters must either be None or equal to rank or depth"

    strides = normalize_strides(depth, rank, tile_rank_over_depth, strides)
    if not hasattr(filters, "__len__"):
        filters = [filters] * depth
    if not hasattr(activations, "__len__") or isinstance(activations, str):
        activations = [activations] * depth

    minimum_output_shape = compute_minimum_output_shape(input_shape, strides)
    assert_minimum_output_shape(minimum_output_shape, target_shape)
    output_shape_margin = compute_output_shape_margin(minimum_output_shape, target_shape)
    kernel_sizes = compute_kernel_sizes(output_shape_margin, strides)

    strides = rank_to_depth_major(strides)

    deconv_layers = []
    for i in range(depth):
        deconv_layer = Conv2DTranspose(filters=filters[i], strides=strides[i], kernel_size=kernel_sizes[i],
                                       activation=activations[i])
        deconv_layers.append(deconv_layer)
    return deconv_layers


def check_parameters_lengths(*args):
    length = None
    for arg in args:
        if hasattr(arg, "__len__") and not isinstance(arg, str):
            if length is None:
                length = len(arg)
            else:
                assert len(arg) == length, "All parameters must have the same length, when provided"
    return length


def normalize_strides(depth: int,
                      rank: int,
                      tile_rank_over_depth: bool,
                      strides):
    """ Normalize strides to output a list with shape [rank, depth],
    where rank is major for convenience. """
    if hasattr(strides, "__len__"):
        if len(strides) == depth:
            if len(strides) == rank:
                if tile_rank_over_depth:
                    normalized_arg = tile_rank(strides, depth)
                else:
                    normalized_arg = tile_depth(strides, rank)
            else:
                normalized_arg = tile_depth(strides, rank)
        else:
            normalized_arg = tile_rank(strides, depth)
    else:
        normalized_arg = tile_rank(tile_depth(strides, rank), depth)

    return normalized_arg


def tile_depth(arg,
               rank: int):
    return [arg for _ in range(rank)]


def tile_rank(arg,
              depth: int):
    rank = len(arg)
    return [[arg[i] for _ in range(depth)]
            for i in range(rank)]


def compute_minimum_output_shape(input_shape, strides):
    shape = []
    for i in range(len(strides)):
        minimum_size = compute_minimum_output_size(input_shape[i], strides[i])
        shape.append(minimum_size)
    return shape


def compute_minimum_output_size(input_size, strides):
    size = input_size
    for i in range(len(strides)):
        size *= strides[i]
    return size


def compute_output_shape_margin(minimum_output_shape, target_shape):
    margin = []
    for i in range(len(minimum_output_shape)):
        margin.append(target_shape[i] - minimum_output_shape[i])
    return margin


def assert_minimum_output_shape(minimum_output_shape, target_shape):
    """ Raise an AssertionError when one of the dimensions of the target shape is lower than
    of the minimum output shape. """
    for i in range(len(minimum_output_shape)):
        assert minimum_output_shape[i] <= target_shape[i], \
            "Target shape is too low given input shape, strides and kernels : " \
            "{0} < {1} (at rank {2})".format(target_shape[i], minimum_output_shape[i], i)


def compute_kernel_sizes(output_shape_margin, strides):
    kernel_sizes = []
    for i in range(len(strides)):
        kernel_sizes.append(compute_kernel_sizes_for_dim(output_shape_margin[i], strides[i]))

    return rank_to_depth_major(kernel_sizes)


def compute_kernel_sizes_for_dim(output_size_margin, strides):
    base = 0
    multiplier = 1
    multipliers = []
    for i in range(len(strides)):
        base += multiplier
        multipliers.append(multiplier)
        multiplier *= strides[len(strides) - i - 2]

    kernel_size_base = output_size_margin // base
    remaining = output_size_margin - kernel_size_base * base
    kernel_size_base = [kernel_size_base] * len(strides)
    for i in range(len(strides)):
        multiplier = multipliers[len(strides) - i - 1]
        if multiplier <= remaining:
            kernel_size_add = remaining // multiplier
            remaining -= multiplier * kernel_size_add
            kernel_size_base[i] += kernel_size_add
        kernel_size_base[i] += strides[i]
    return kernel_size_base


def rank_to_depth_major(rank_major):
    depth = len(rank_major[0])
    rank = len(rank_major)
    depth_major = []
    for i in range(depth):
        depth_elements = []
        for j in range(rank):
            depth_elements.append(rank_major[j][i])
        depth_major.append(depth_elements)
    return depth_major
