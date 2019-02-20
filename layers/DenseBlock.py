from keras.layers import Layer, InputSpec, BatchNormalization
from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints
from keras import backend as K


class _DenseBlock(Layer):
    def __init__(self, rank,
                 kernel_size,
                 growth_rate,
                 depth,
                 use_bottleneck=True,
                 bottleneck_filters_multiplier=4,
                 use_batch_normalization=True,
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        assert rank in [1, 2, 3]

        super(_DenseBlock, self).__init__(**kwargs)

        self.rank = rank
        self.conv = K.conv1d if rank is 1 else K.conv2d if rank is 2 else K.conv3d
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.growth_rate = growth_rate

        if use_bottleneck:
            assert (depth % 2) == 0, "Depth must be a multiple of 2 when using bottlenecks."
        self._depth = depth // 2 if use_bottleneck else depth
        self.use_bottleneck = use_bottleneck

        self.bottleneck_filters = bottleneck_filters_multiplier * self.growth_rate
        self.bottleneck_kernel_size = conv_utils.normalize_tuple(1, rank, "bottleneck_kernel_size")
        self.use_batch_normalization = use_batch_normalization

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

        self.kernels = []
        self.biases = []
        self.bottleneck_kernels = []
        self.bottleneck_biases = []

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_dim = self.get_input_dim(input_shape)
        total_dim = input_dim

        with K.name_scope("dense_block_weights"):
            for i in range(self._depth):
                with K.name_scope("layer_weights_{}".format(i)):
                    if self.use_bottleneck:
                        self.add_kernel_weights_for(total_dim, bottleneck=True)
                        if self.use_bias:
                            self.add_bias_weights_for(bottleneck=True)
                        input_dim = self.bottleneck_filters
                    else:
                        input_dim = total_dim

                    self.add_kernel_weights_for(input_dim, bottleneck=False)
                    if self.use_bias:
                        self.add_bias_weights_for(bottleneck=False)

                    total_dim += self.growth_rate

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: self.get_input_dim(input_shape)})
        super(_DenseBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [inputs]
        for i in range(self._depth):
            intermediate_output = K.concatenate(outputs, axis=-1)

            intermediate_output = self.composite_function(intermediate_output, i)

            outputs.append(intermediate_output)

        outputs = K.concatenate(outputs, axis=-1)
        return outputs

    def composite_function(self, inputs, index):
        intermediate_output = inputs

        # region Bottleneck
        if self.use_bottleneck:
            if self.use_batch_normalization:
                intermediate_output = BatchNormalization()(intermediate_output)

            if self.activation is not None:
                intermediate_output = self.activation(intermediate_output)

            intermediate_output = self.conv(intermediate_output, self.bottleneck_kernels[index],
                                            padding="same", data_format=self.data_format)

            if self.use_bias:
                intermediate_output = K.bias_add(intermediate_output,
                                                 self.bottleneck_biases[index],
                                                 data_format=self.data_format)
        # endregion

        if self.use_batch_normalization:
            intermediate_output = BatchNormalization()(intermediate_output)

        if self.activation is not None:
            intermediate_output = self.activation(intermediate_output)

        intermediate_output = self.conv(intermediate_output, self.kernels[index],
                                        padding="same", data_format=self.data_format)

        if self.use_bias:
            intermediate_output = K.bias_add(intermediate_output,
                                             self.biases[index],
                                             data_format=self.data_format)

        return intermediate_output

    def compute_output_shape(self, input_shape):
        input_dim = self.get_input_dim(input_shape)
        output_dim = input_dim + self._depth * self.growth_rate
        output_shape = list(input_shape)
        output_shape[self.channel_axis] = output_dim
        return tuple(output_shape)

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

    def get_kernel_shape(self, input_dim, bottleneck: bool):
        if bottleneck:
            kernel_size = self.bottleneck_kernel_size
            filters = self.bottleneck_filters
        else:
            kernel_size = self.kernel_size
            filters = self.growth_rate

        return kernel_size + (input_dim, filters)

    def add_kernel_weights(self, shape, index):
        return self.add_weight(shape=shape,
                               initializer=self.kernel_initializer,
                               name="kernel_{0}".format(index),
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)

    def add_kernel_weights_for(self, input_dim, bottleneck: bool):
        shape = self.get_kernel_shape(input_dim, bottleneck)
        if bottleneck:
            kernel_weights = self.add_kernel_weights(shape, len(self.bottleneck_kernels))
            self.bottleneck_kernels.append(kernel_weights)
        else:
            kernel_weights = self.add_kernel_weights(shape, len(self.kernels))
            self.kernels.append(kernel_weights)

    def add_bias_weights(self, filters, index):
        return self.add_weight(shape=(filters,),
                               initializer=self.bias_initializer,
                               name="bias_{0}".format(index),
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)

    def add_bias_weights_for(self, bottleneck: bool):
        if bottleneck:
            bias_weights = self.add_bias_weights(self.bottleneck_filters, len(self.bottleneck_biases))
            self.bottleneck_biases.append(bias_weights)
        else:
            bias_weights = self.add_bias_weights(self.growth_rate, len(self.biases))
            self.biases.append(bias_weights)

    @property
    def depth(self):
        return self._depth * 2 if self.use_bottleneck else self._depth

    def get_config(self):
        config = \
            {
                "rank": self.rank,
                "kernel_size": self.kernel_size,
                "growth_rate": self.growth_rate,
                "depth": self.depth,
                "use_bottleneck": self.use_bottleneck,
                "use_batch_normalization": self.use_batch_normalization,
                "data_format": self.data_format,
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
        base_config = super(_DenseBlock, self).get_config()
        return {**base_config, **config}


def evaluate_on_cifar10():
    from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Dropout, Reshape, ReLU, BatchNormalization
    from keras.models import Model
    from keras.datasets import cifar10
    from keras.utils.np_utils import to_categorical
    import numpy as np

    input_layer = Input(shape=[32, 32, 3])

    total_depth = 16
    n_blocks = 3
    depth = (total_depth - 4) // n_blocks
    _growth_rate = 2
    n_channels = _growth_rate * 2

    layer = input_layer
    layer = Conv2D(filters=n_channels, kernel_size=3, strides=1, padding="same")(layer)

    for k in range(n_blocks):
        layer = _DenseBlock(rank=2, kernel_size=3, growth_rate=_growth_rate, depth=depth)(layer)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)

        if k < (n_blocks - 1):
            n_channels *= 2
            layer = Conv2D(filters=n_channels, kernel_size=1)(layer)
            layer = Dropout(rate=0.5)(layer)
            layer = AveragePooling2D(pool_size=2, strides=2)(layer)
        else:
            layer = AveragePooling2D(pool_size=8)(layer)

    layer = Reshape([-1])(layer)
    layer = Dense(units=10, activation="softmax")(layer)
    model = Model(inputs=input_layer, outputs=layer)
    model.summary()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    model.fit(x=x_train, y=y_train, validation_data=[x_test, y_test], epochs=300, batch_size=256)


if __name__ == "__main__":
    evaluate_on_cifar10()
