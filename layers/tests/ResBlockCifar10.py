from keras.layers import Input, AveragePooling2D, Dense, Reshape
from keras.models import Model
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from layers import ResBlock2D


def evaluate_on_cifar10():
    total_depth = 36
    n_blocks = 3
    depth = total_depth // n_blocks

    # region Model
    input_layer = Input(shape=[32, 32, 3])
    layer = input_layer

    for k in range(n_blocks):
        for i in range(depth):
            layer = ResBlock2D(filters=16*(2**k), kernel_size=3, use_batch_normalization=False)(layer)

        if k < (n_blocks - 1):
            layer = AveragePooling2D(pool_size=2, strides=2)(layer)
        else:
            layer = AveragePooling2D(pool_size=8)(layer)

    layer = Reshape([-1])(layer)
    layer = Dense(units=10, activation="softmax")(layer)
    model = Model(inputs=input_layer, outputs=layer)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    # endregion

    # region Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)
    generator.fit(x_train, seed=0)
    # endregion

    model.fit_generator(generator.flow(x_train, y_train, batch_size=100),
                        steps_per_epoch=100, epochs=300, validation_data=(x_test, y_test),
                        validation_steps=100, verbose=1)


if __name__ == "__main__":
    evaluate_on_cifar10()
