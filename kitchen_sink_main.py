import tensorflow as tf
import numpy as np


def main():
    tf.random.set_seed(42)

    x = tf.random.normal(shape=[1024, 32])
    x = x.numpy()
    # np.save(r"D:\Users\Degva\Desktop\tmp\work\tf_random_seed_test.npy", x)
    y = np.load(r"D:\Users\Degva\Desktop\tmp\work\tf_random_seed_test.npy")

    z = x - y
    print(z.mean())


if __name__ == "__main__":
    main()
