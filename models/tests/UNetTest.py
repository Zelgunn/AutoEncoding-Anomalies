from keras.models import Model
from keras.layers import Input, AveragePooling3D, UpSampling3D, concatenate, Conv3D, ConvLSTM2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import cv2
import numpy as np
from time import time
import os

from layers import DenseBlock3D
from datasets import SubwayDatabase


def get_conv_model():
    input_layer = Input(shape=[8, 96, 128, 1])
    layer = input_layer

    encoder_layers = []

    n = 16

    layer = Conv3D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = Conv3D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = Conv3D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = Conv3D(filters=n * 8, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 8, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=(1, 2, 2))(layer)

    layer = Conv3D(filters=n * 16, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)

    layer = UpSampling3D(size=(1, 2, 2))(layer)
    layer = Conv3D(filters=n * 8, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 8, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = concatenate([encoder_layers[3], layer])

    layer = UpSampling3D(size=2)(layer)
    layer = Conv3D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = concatenate([encoder_layers[2], layer])

    layer = UpSampling3D(size=2)(layer)
    layer = Conv3D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = concatenate([encoder_layers[1], layer])

    layer = UpSampling3D(size=2)(layer)
    layer = Conv3D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = Conv3D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same")(layer)
    layer = concatenate([encoder_layers[0], layer])

    output_layer = Conv3D(filters=1, kernel_size=1, kernel_initializer="he_normal", activation="sigmoid",
                          use_bias=False)(layer)

    conv_block_model = Model(inputs=input_layer, outputs=output_layer)
    conv_block_model.summary(line_length=200)
    return conv_block_model


def get_conv_lstm_model():
    input_layer = Input(shape=[8, 96, 128, 1])
    layer = input_layer

    encoder_layers = []

    n = 16

    layer = ConvLSTM2D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same",
                       return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal", padding="same",
                       return_sequences=True)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = ConvLSTM2D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = ConvLSTM2D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)

    layer = ConvLSTM2D(filters=n * 8, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)

    layer = UpSampling3D(size=2)(layer)
    layer = ConvLSTM2D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n * 4, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = concatenate([encoder_layers[2], layer])

    layer = UpSampling3D(size=2)(layer)
    layer = ConvLSTM2D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n * 2, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = concatenate([encoder_layers[1], layer])

    layer = UpSampling3D(size=2)(layer)
    layer = ConvLSTM2D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = ConvLSTM2D(filters=n, kernel_size=3, use_bias=False, kernel_initializer="he_normal",
                       padding="same", return_sequences=True)(layer)
    layer = concatenate([encoder_layers[0], layer])

    output_layer = ConvLSTM2D(filters=1, kernel_size=1, kernel_initializer="he_normal", activation="sigmoid",
                              use_bias=False, return_sequences=True)(layer)

    conv_lstm_model = Model(inputs=input_layer, outputs=output_layer)
    conv_lstm_model.summary(line_length=200)
    return conv_lstm_model


def get_dense_block_model():
    input_layer = Input(shape=[8, 96, 128, 1])
    layer = input_layer

    encoder_layers = []
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=16, use_bias=False)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=32, use_bias=False)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=64, use_bias=False)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=128, use_bias=False)(layer)
    encoder_layers.append(layer)
    layer = AveragePooling3D(pool_size=(1, 2, 2))(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=256, use_bias=False)(layer)

    layer = UpSampling3D(size=(1, 2, 2))(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=128, use_bias=False)(layer)
    layer = concatenate([encoder_layers[3], layer])
    layer = UpSampling3D(size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=64, use_bias=False)(layer)
    layer = concatenate([encoder_layers[2], layer])
    layer = UpSampling3D(size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=32, use_bias=False)(layer)
    layer = concatenate([encoder_layers[1], layer])
    layer = UpSampling3D(size=2)(layer)
    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=8, output_filters=16, use_bias=False)(layer)
    layer = concatenate([encoder_layers[0], layer])

    output_layer = Conv3D(filters=1, kernel_size=1, kernel_initializer="he_normal", activation="sigmoid",
                          use_bias=False)(layer)

    dense_block_model = Model(inputs=input_layer, outputs=output_layer)
    dense_block_model.summary(line_length=200)
    return dense_block_model


def get_database():
    subway_database = SubwayDatabase(database_path="../datasets/subway/exit",
                                     input_sequence_length=None,
                                     output_sequence_length=None)
    subway_database.load()

    subway_database = subway_database.resized(image_size=(96, 128), input_sequence_length=8, output_sequence_length=8)
    subway_database.normalize(0.0, 1.0)

    subway_database.train_dataset.epoch_length = 250
    subway_database.train_dataset.batch_size = 16
    subway_database.test_dataset.epoch_length = 25
    subway_database.test_dataset.batch_size = 2

    return subway_database


def main():
    model = get_conv_lstm_model()
    database = get_database()

    model.compile(optimizer=Adam(lr=1e-4), loss="mae", metrics=["mse"])
    # model.load_weights("../logs/tests/1551101991.9433231/weights.h5")

    log_dir = "../logs/tests/{}".format(time())
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit_generator(database.train_dataset, epochs=2, validation_data=database.test_dataset,
                        callbacks=[tensorboard])

    key = 13
    escape_key = 27
    seed = 0
    i = 0

    y_pred, y_true, x = None, None, None

    while key != escape_key:
        if key != -1:
            seed += 1

            x, y_true = database.test_dataset.get_batch(seed=seed)
            y_pred = model.predict(x)
            i = 0

        y_pred_resized = cv2.resize(y_pred[0][i], dsize=(256, 192))
        y_true_resized = cv2.resize(y_true[0][i], dsize=(256, 192))
        x_resized = cv2.resize(x[0][i], dsize=(256, 192))

        cv2.imshow("inputs", x_resized)
        cv2.imshow("y_true", y_true_resized)
        cv2.imshow("y_pred", y_pred_resized)
        cv2.imshow("delta", np.abs(y_pred_resized - y_true_resized))
        key = cv2.waitKey(100)

        i = (i + 1) % 8

    model.save_weights(os.path.join(log_dir, "weights.h5"))


if __name__ == "__main__":
    main()
