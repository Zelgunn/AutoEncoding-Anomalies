from keras.layers import Layer, InputSpec, BatchNormalization
from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints
from keras import backend as K


class _DenseBlock(Layer):
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 growth_rate,
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_batch_normalization=True,
                 **kwargs):

        super(_DenseBlock, self).__init__(**kwargs)

        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.growth_rate = growth_rate

        self.data_format = K.normalize_data_format(data_format)
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

        self.strides = conv_utils.normalize_tuple(1, rank, "strides")
        self.padding = "same"
        self.dilation_rate = conv_utils.normalize_tuple(1, rank, "dilation_rate")

        self.kernels = []
        self.projection_kernel = None
        self.biases = []

        self.input_spec = InputSpec(ndim=self.rank + 2)
