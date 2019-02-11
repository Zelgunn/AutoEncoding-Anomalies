from keras.layers import Layer, InputSpec, BatchNormalization
from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints
from keras import backend as K

RES_DEPTH = 2


class _ResBlock(Layer):
    def __init__(self, rank,
                 filters,
                 kernel_size,
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
                 use_batch_normalization=False,
                 **kwargs):

        super(_ResBlock, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.projection_kernel_size = conv_utils.normalize_tuple(1, rank, "projection_kernel_size")
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

        self.kernels = []
        self.projection_kernel = None
        self.biases = []

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_dim = self.get_input_dim(input_shape)
        for i in range(RES_DEPTH):
            self.add_kernel(input_dim if (i == 0) else self.filters)
            if self.use_bias:
                self.add_bias()

        if self.use_projection(input_shape):
            self.projection_kernel = self.add_weight(shape=self.get_kernel_shape(input_dim, True),
                                                     initializer=self.kernel_initializer,
                                                     name="projection_kernel",
                                                     regularizer=self.kernel_regularizer,
                                                     constraint=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={self.channel_axis: self.get_input_dim(input_shape)})
        super(_ResBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        conv = None
        if self.rank == 1:
            conv = K.conv1d
        elif self.rank == 2:
            conv = K.conv2d
        elif self.rank == 3:
            conv = K.conv3d

        outputs = inputs
        for i in range(RES_DEPTH):
            strides = self.get_strides_at_depth(i)

            if self.use_batch_normalization:
                outputs = BatchNormalization()(outputs)

            if self.activation is not None:
                outputs = self.activation(outputs)

            outputs = conv(
                outputs,
                self.kernels[i],
                strides=strides,
                padding="same",
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

            if self.use_bias:
                outputs = K.bias_add(
                    outputs,
                    self.biases[i],
                    data_format=self.data_format)

        if self.use_projection(K.int_shape(inputs)):
            inputs = conv(inputs, self.projection_kernel, strides=self.strides, padding="same")

        # x_k+1 = x_k + f(x_k)
        outputs = K.sum([outputs, inputs], axis=0)
        return outputs

    def use_projection(self, input_shape):
        assert None not in input_shape[1:], "Cannot decide if projection is needed without knowing input shape (TO FIX)"
        return input_shape != self.compute_output_shape(input_shape)

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
        if input_shape[self.channel_axis] is None:
            raise ValueError("The channel dimension of the inputs "
                             "should be defined. Found `None`.")
        return input_shape[self.channel_axis]

    def get_kernel_shape(self, input_dim, for_projection_kernel=False):
        kernel_size = self.kernel_size if not for_projection_kernel else self.projection_kernel_size
        return kernel_size + (input_dim, self.filters)

    def add_kernel(self, input_dim):
        kernel = self.add_weight(shape=self.get_kernel_shape(input_dim),
                                 initializer=self.kernel_initializer,
                                 name="kernel_{0}".format(len(self.kernels)),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.kernels += [kernel]

    def add_bias(self):
        bias = self.add_weight(shape=(self.filters,),
                               initializer=self.bias_initializer,
                               name="bias_{0}".format(len(self.biases)),
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)
        self.biases += [bias]

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
                "depth": RES_DEPTH,
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
                                         filters=filters, kernel_size=kernel_size,
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
                                         filters=filters, kernel_size=kernel_size,
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
        super(ResBlock3D, self).__init__(rank=3,
                                         filters=filters, kernel_size=kernel_size,
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
    def get_kernel_shape(self, input_dim, for_projection_kernel=False):
        kernel_size = self.kernel_size if not for_projection_kernel else self.projection_kernel_size
        return kernel_size + (self.filters, input_dim)

    def call(self, inputs, **kwargs):
        deconv = None
        if self.rank == 2:
            deconv = K.conv2d_transpose
        elif self.rank == 3:
            deconv = K.conv3d_transpose

        outputs = inputs
        projection_output_shape = None
        for i in range(RES_DEPTH):
            strides = self.get_strides_at_depth(i)

            if self.use_batch_normalization:
                outputs = BatchNormalization()(outputs)

            if self.activation is not None:
                outputs = self.activation(outputs)

            deconv_output_shape = self.get_deconv_output_shape(outputs, i)
            if i == 0:
                projection_output_shape = deconv_output_shape

            outputs = deconv(
                outputs,
                self.kernels[i],
                deconv_output_shape,
                strides=strides,
                padding="same",
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

            if self.use_bias:
                outputs = K.bias_add(
                    outputs,
                    self.biases[i],
                    data_format=self.data_format)

        if self.use_projection(K.int_shape(inputs)):
            inputs = deconv(inputs, self.projection_kernel, projection_output_shape,
                            strides=self.strides, padding="same")

        # x_k+1 = x_k + f(x_k)
        outputs = K.sum([outputs, inputs], axis=0)
        return outputs

    def get_deconv_output_shape(self, inputs, depth):
        strides = self.get_strides_at_depth(depth)

        input_shape = K.shape(inputs)
        batch_size = input_shape[0]

        if self.channels_first:
            spatial_axis = list(range(2, self.rank + 2))
        else:
            spatial_axis = list(range(1, self.rank + 1))

        inputs_spatial_dims = [input_shape[axis] for axis in spatial_axis]
        out_spatial_dims = []
        for i in range(len(inputs_spatial_dims)):
            out_dim = conv_utils.deconv_length(inputs_spatial_dims[i], strides[i], self.kernel_size[i],
                                               padding="same", output_padding=None, dilation=self.dilation_rate[i])
            out_spatial_dims.append(out_dim)

        if self.channels_first:
            output_shape = (batch_size, self.filters, *out_spatial_dims)
        else:
            output_shape = (batch_size, *out_spatial_dims, self.filters)
        return output_shape

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.deconv_length(
                    space[i],
                    self.strides[i],
                    self.kernel_size[i],
                    padding="same",
                    output_padding=None,
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)


class ResBlock2DTranspose(_ResBlockTranspose):
    def __init__(self, filters,
                 kernel_size,
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
        super(ResBlock2DTranspose, self).__init__(rank=2,
                                                  filters=filters, kernel_size=kernel_size,
                                                  strides=strides, data_format=data_format,
                                                  dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
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
        super(ResBlock3DTranspose, self).__init__(rank=3,
                                                  filters=filters, kernel_size=kernel_size,
                                                  strides=strides, data_format=data_format,
                                                  dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
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
