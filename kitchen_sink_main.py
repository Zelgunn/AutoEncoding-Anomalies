from kitchen_sink.audioset_ebl3 import main

import numpy as np
import matplotlib.pyplot as plt


def test(scale):
    dec = 3

    x = np.random.normal(loc=1.0, scale=scale, size=[2 ** 18])
    x = np.clip(x, 0.0, 2.0)
    y = 2.0 - x

    cond = x < 1.0
    z = np.where(cond, x, y)
    z = np.round(z, decimals=dec) * 10 ** dec
    z = z.astype(np.int32)
    z = np.clip(z, 0, 10 ** dec - 1)

    h = np.zeros(shape=[10 ** dec], dtype=np.int32)

    for value in z:
        h[value] += 1

    return h


def tmp_main():
    for i in range(2):
        h = test(0.25 / (i + 1))
        plt.plot(h)

    plt.show()


if __name__ == "__main__":
    tmp_main()
    exit()
    main()
