import tensorflow as tf
import timeit
from tensorflow.python.keras.layers import Conv3D, Dense

from CustomKerasLayers.utility_layers.Unfold import Unfold
from CustomKerasLayers.layers.StandAloneSelfAttention import StandAloneSelfAttention3D


def test_stride():
    filters = 32
    head_count = 8
    head_size = filters // head_count
    k = 4
    # kernel_size = (k, k, k)
    x = tf.random.normal(shape=(8, 16, 32, 32, 1))

    sasa_layer_1 = StandAloneSelfAttention3D(head_size, head_count, kernel_size=(2, 2, 2), strides=(2, 2, 2))
    sasa_layer_1(x)

    sasa_layer_2 = StandAloneSelfAttention3D(head_size, head_count, kernel_size=(4, 4, 4), strides=(2, 2, 2))
    sasa_layer_2(x)

    @tf.function
    def test_sasa_1(_x):
        return sasa_layer_1(_x)

    @tf.function
    def test_sasa_2(_x):
        return sasa_layer_2(_x)

    test_sasa_1(x)
    test_sasa_2(x)

    print("SASA (1):", timeit.timeit(lambda: test_sasa_1(x), number=1000))
    print("SASA (2):", timeit.timeit(lambda: test_sasa_2(x), number=1000))


def test_vs():
    filters = 32
    head_count = 8
    # head_size = filters // head_count
    k = 4
    kernel_size = (k, k, k)
    x = tf.random.normal(shape=(1, 8, 16, 16, 1))

    # unfold_layer = Unfold(kernel_size=kernel_size, strides=kernel_size, padding="SAME")

    sasa_layer_v1 = Dense(filters)
    sasa_layer_v1.build((8, 16, 16, 1))
    sasa_layer_v1(x)

    # sasa_layer_v2 = StandAloneSelfAttention3D(head_size, head_count, kernel_size=kernel_size, strides=kernel_size,
    #                                           v1=False)
    # sasa_layer_v2.build((32, 128, 128, 1))
    # sasa_layer_v2(x)

    conv_layer = Conv3D(filters=filters, kernel_size=kernel_size, strides=kernel_size, padding="SAME")
    conv_layer.build((8, 16, 16, 1))

    # @tf.function
    # def test_unfold(_x):
    #     return unfold_layer(_x)
    #
    # @tf.function
    # def test_sasa_v1(_x):
    #     return sasa_layer_v1(_x)
    #
    # @tf.function
    # def test_sasa_v2(_x):
    #     return sasa_layer_v2(_x)

    @tf.function
    def test_conv(_x):
        return conv_layer(_x)

    # test_unfold(x)
    # test_sasa_v1(x)
    # test_sasa_v2(x)
    test_conv(x)

    # print("Unfold:", timeit.timeit(lambda: test_unfold(x), number=100))
    # print("SASA v1:", timeit.timeit(lambda: test_sasa_v1(x), number=100))
    # print("SASA v2:", timeit.timeit(lambda: test_sasa_v2(x), number=100))
    print("Conv:", timeit.timeit(lambda: test_conv(x), number=100))


def main():
    test_vs()


if __name__ == "__main__":
    main()
