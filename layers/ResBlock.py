from keras.layers import InputSpec, BatchNormalization
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import Deconv2D, Deconv3D
from keras.utils import conv_utils
from keras.utils.generic_utils import to_list
from keras import activations, initializers, regularizers, constraints
from keras import backend as K

from layers import CompositeLayer


class _ResBlock(CompositeLayer):
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):

        assert rank in [1, 2, 3]
        assert depth > 0

        super(_ResBlock, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.depth = depth
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.use_batch_normalization = use_batch_normalization

        self.conv_layers = []
        self.batch_normalization_layers = []
        self.projection_layer = None

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def get_conv_layer_type(self):
        return Conv1D if self.rank is 1 else Conv2D if self.rank is 2 else Conv3D

    def init_layers(self, input_shape):
        conv_layer_type = self.get_conv_layer_type()
        for i in range(self.depth):
            strides = self.get_strides_at_depth(i)
            conv_layer = conv_layer_type(filters=self.filters,
                                         kernel_size=self.kernel_size,
                                         strides=strides,
                                         padding="same",
                                         data_format=self.data_format,
                                         dilation_rate=self.dilation_rate,
                                         use_bias=self.use_bias,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer,
                                         kernel_regularizer=self.kernel_regularizer,
                                         bias_regularizer=self.bias_regularizer,
                                         activity_regularizer=self.activity_regularizer,
                                         kernel_constraint=self.kernel_constraint,
                                         bias_constraint=self.bias_constraint)
            self.conv_layers.append(conv_layer)

            if self.use_batch_normalization:
                self.batch_normalization_layers.append(BatchNormalization())

        if self.use_projection(input_shape):
            projection_kernel_size = conv_utils.normalize_tuple(1, self.rank, "projection_kernel_size")
            self.projection_layer = conv_layer_type(filters=self.filters,
                                                    kernel_size=projection_kernel_size,
                                                    strides=self.strides,
                                                    padding="same",
                                                    data_format=self.data_format,
                                                    dilation_rate=self.dilation_rate,
                                                    use_bias=False,
                                                    kernel_initializer=self.kernel_initializer,
                                                    bias_initializer=self.bias_initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    bias_regularizer=self.bias_regularizer,
                                                    activity_regularizer=self.activity_regularizer,
                                                    kernel_constraint=self.kernel_constraint,
                                                    bias_constraint=self.bias_constraint)

    def build(self, input_shape):
        self.init_layers(input_shape)
        intermediate_shape = input_shape

        with K.name_scope("residual_block_weights"):
            for i in range(self.depth):
                self.build_sub_layer(self.conv_layers[i], intermediate_shape)
                intermediate_shape = self.conv_layers[i].compute_output_shape(intermediate_shape)

            if self.projection_layer is not None:
                self.build_sub_layer(self.projection_layer, input_shape)

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]})
        super(_ResBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        for i in range(self.depth):
            with K.name_scope("residual_layer_{}".format(i)):
                if self.use_batch_normalization:
                    outputs = self.batch_normalization_layers[i](outputs)

                if self.activation is not None:
                    outputs = self.activation(outputs)

                outputs = self.conv_layers[i](outputs)

        if self.use_projection(K.int_shape(inputs)):
            inputs = self.projection_layer(inputs)

        # x_k+1 = x_k + f(x_k)
        outputs = K.sum([outputs, inputs], axis=0)
        return outputs

    def use_projection(self, input_shape):
        strides = to_list(self.strides, allow_tuple=True)
        for stride in strides:
            if stride != 1:
                return True

        return input_shape[self.channel_axis] != self.filters

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding="same",
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    def get_input_dim(self, input_shape):
        assert input_shape[self.channel_axis] is not None, \
            "The channel dimension of the inputs should be defined. Found `None`."
        return input_shape[self.channel_axis]

    def get_strides_at_depth(self, depth):
        if depth == 0:
            strides = self.strides
        else:
            strides = conv_utils.normalize_tuple(1, self.rank, "strides")
        return strides

    def get_config(self):
        config = \
            {
                "rank": self.rank,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "depth": self.depth,
                "strides": self.strides,
                "padding": "same",
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint)
            }
        base_config = super(_ResBlock, self).get_config()
        return {**base_config, **config}


class ResBlock1D(_ResBlock):
    def __init__(self, filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):
        super(ResBlock1D, self).__init__(rank=1,
                                         filters=filters, kernel_size=kernel_size, depth=depth,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         use_batch_normalization=use_batch_normalization,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock1D, self).get_config()
        config.pop("rank")
        return config


class ResBlock2D(_ResBlock):
    def __init__(self, filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):
        super(ResBlock2D, self).__init__(rank=2,
                                         filters=filters, kernel_size=kernel_size, depth=depth,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         use_batch_normalization=use_batch_normalization,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock2D, self).get_config()
        config.pop("rank")
        return config


class ResBlock3D(_ResBlock):
    def __init__(self, filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):
        super(ResBlock3D, self).__init__(rank=3,
                                         filters=filters, kernel_size=kernel_size, depth=depth,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         use_batch_normalization=use_batch_normalization,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock3D, self).get_config()
        config.pop("rank")
        return config


class _ResBlockTranspose(_ResBlock):
    def get_conv_layer_type(self):
        assert self.rank in [2, 3]
        return Deconv2D if self.rank is 2 else Deconv3D

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.deconv_length(
                    space[i],
                    self.strides[i],
                    self.kernel_size[i],
                    padding="same",
                    output_padding=None)
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)


class ResBlock2DTranspose(_ResBlockTranspose):
    def __init__(self, filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1),
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):
        super(ResBlock2DTranspose, self).__init__(rank=2,
                                                  filters=filters, kernel_size=kernel_size, depth=depth,
                                                  strides=strides, data_format=data_format,
                                                  activation=activation, use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                  use_batch_normalization=use_batch_normalization,
                                                  **kwargs)

    def get_config(self):
        config = super(ResBlock2DTranspose, self).get_config()
        config.pop("rank")
        return config


class ResBlock3DTranspose(_ResBlockTranspose):
    def __init__(self, filters,
                 kernel_size,
                 depth=2,
                 strides=(1, 1, 1),
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):
        super(ResBlock3DTranspose, self).__init__(rank=3,
                                                  filters=filters, kernel_size=kernel_size, depth=depth,
                                                  strides=strides, data_format=data_format,
                                                  activation=activation, use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                  use_batch_normalization=use_batch_normalization,
                                                  **kwargs)

    def get_config(self):
        config = super(ResBlock3DTranspose, self).get_config()
        config.pop("rank")
        return config
