from kitchen_sink.audioset_ebl3 import main
import tensorflow as tf
from time import time


def tmp_main():
    x = tf.Variable(initial_value=0.1)
    y = tf.Variable(initial_value=1.0)

    with tf.GradientTape() as tape:
        z = x * y
        loss = tf.abs(z + 0.5)

    print(tape.gradient(loss, [x, y]))


def compare_einsum_matmul():
    batch_size = 8
    count = 16 ** 2
    heads = 8
    head_size = 16
    k = 3

    x = tf.random.normal(shape=[batch_size, heads, head_size, k ** 2, count])
    y = tf.random.normal(shape=[batch_size, count, heads, 1, k ** 2])

    t0 = time()
    for _ in range(100):
        _ = test_einsum(x, y)

    t1 = time()
    print("einsum : {}".format(round(t1 - t0, 2)))

    for _ in range(100):
        _ = test_matmul_transpose(x, y)

    t2 = time()
    print("matmul_transpose : {}".format(round(t2 - t1, 2)))


@tf.function
def test_einsum(x, y):
    # noinspection SpellCheckingInspection
    return tf.einsum("bhskc,bchsk->bhsc", x, y)


@tf.function
def test_matmul_transpose(x, y):
    x = tf.transpose(x, perm=[0, ])


if __name__ == "__main__":
    tmp_main()
    exit()
    main()
