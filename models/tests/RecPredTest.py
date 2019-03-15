from keras.models import Model
from keras.layers import Input, Conv3D, Deconv3D, BatchNormalization, LeakyReLU, MaxPooling3D, concatenate, UpSampling3D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import os
from time import time

from layers import ResBasicBlock3D, DenseBlock3D, ResBasicBlock3DTranspose
from datasets import UCSDDatabase, SubwayDatabase
from data_preprocessors import BrightnessShifter, DropoutNoiser
from utils.test_utils import visualize_model_errors, evaluate_model_anomaly_detection_on_subset


def build_conv_encoder(input_layer):
    layer = input_layer

    layer = Conv3D(kernel_size=3, filters=32, kernel_initializer="he_normal", padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = Conv3D(kernel_size=3, filters=48, kernel_initializer="he_normal", padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = Conv3D(kernel_size=3, filters=64, kernel_initializer="he_normal", padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = Conv3D(kernel_size=3, filters=64, kernel_initializer="he_normal", padding="same")(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    return layer


def build_residual_block_encoder(input_layer):
    layer = input_layer

    layer = ResBasicBlock3D(kernel_size=3, filters=32, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBasicBlock3D(kernel_size=3, filters=48, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBasicBlock3D(kernel_size=3, filters=64, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBasicBlock3D(kernel_size=3, filters=64, kernel_initializer="he_normal")(layer)

    return layer


def build_dense_block_encoder(input_layer):
    layer = input_layer

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=32)(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=48)(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=64)(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=64)(layer)

    return layer


def build_conv_decoder(input_layer):
    layer = input_layer

    layer = Deconv3D(kernel_size=3, filters=48, kernel_initializer="he_normal", padding="same", strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    layer = Deconv3D(kernel_size=3, filters=32, kernel_initializer="he_normal", padding="same", strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    layer = Deconv3D(kernel_size=3, filters=32, kernel_initializer="he_normal", padding="same", strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    layer = Conv3D(kernel_size=1, filters=1, kernel_initializer="he_normal", activation="sigmoid")(layer)

    return layer


def build_residual_block_decoder(input_layer):
    layer = input_layer

    layer = ResBasicBlock3DTranspose(kernel_size=3, filters=48, kernel_initializer="he_normal", strides=2)(layer)
    layer = ResBasicBlock3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", strides=2)(layer)
    layer = ResBasicBlock3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", strides=2)(layer)

    layer = Conv3D(kernel_size=1, filters=1, kernel_initializer="he_normal", activation="sigmoid")(layer)

    return layer


def build_dense_block_decoder(input_layer):
    layer = input_layer

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=48)(layer)
    layer = UpSampling3D(size=2)(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=32)(layer)
    layer = UpSampling3D(size=2)(layer)

    layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=32)(layer)
    layer = UpSampling3D(size=2)(layer)

    layer = Conv3D(kernel_size=1, filters=1, kernel_initializer="he_normal", activation="sigmoid")(layer)

    return layer


def get_model(mode):
    input_layer = Input([16, 128, 128, 1])

    if mode == "residual_block":
        encoder_output_layer = build_residual_block_encoder(input_layer)
        reconstructor_output_layer = build_residual_block_decoder(encoder_output_layer)
        predictor_output_layer = build_residual_block_decoder(encoder_output_layer)
    elif mode == "dense_block":
        encoder_output_layer = build_dense_block_encoder(input_layer)
        reconstructor_output_layer = build_dense_block_decoder(encoder_output_layer)
        predictor_output_layer = build_dense_block_decoder(encoder_output_layer)
    else:
        encoder_output_layer = build_conv_encoder(input_layer)
        reconstructor_output_layer = build_conv_decoder(encoder_output_layer)
        predictor_output_layer = build_conv_decoder(encoder_output_layer)

    output_layer = concatenate([reconstructor_output_layer, predictor_output_layer], axis=1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary(line_length=200)
    exit()
    return model


def get_ucsd_database(shift_brightness, dropout_noise):
    train_preprocessors = []
    if shift_brightness:
        train_preprocessors.append(BrightnessShifter(bias=0.2))
    if dropout_noise:
        train_preprocessors.append(DropoutNoiser(0.2, 0.0))

    database = UCSDDatabase(database_path="../datasets/ucsd/ped2",
                            input_sequence_length=None,
                            output_sequence_length=None,
                            train_preprocessors=train_preprocessors)
    database.load()

    database = database.resized(image_size=(128, 128), input_sequence_length=16, output_sequence_length=32)
    database.normalize(0.0, 1.0)

    database.train_subset.epoch_length = 250
    database.train_subset.batch_size = 8
    database.test_subset.epoch_length = 25
    database.test_subset.batch_size = 2

    return database


def get_subway_database(shift_brightness, dropout_noise):
    train_preprocessors = []
    if shift_brightness:
        train_preprocessors.append(BrightnessShifter(bias=0.2))
    if dropout_noise:
        train_preprocessors.append(DropoutNoiser(0.2, 0.0))

    database = SubwayDatabase(database_path="../datasets/subway/exit",
                              input_sequence_length=None,
                              output_sequence_length=None,
                              train_preprocessors=train_preprocessors)
    database.load()

    database = database.resized(image_size=(96, 128), input_sequence_length=16, output_sequence_length=32)
    database.normalize(0.0, 1.0)

    database.train_subset.epoch_length = 250
    database.train_subset.batch_size = 8
    database.test_subset.epoch_length = 25
    database.test_subset.batch_size = 2

    return database


def train_model(model, database, mode):
    log_dir = "../logs/tests/conv3d_rec_pred/{}/{}".format(mode, int(time()))
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq=database.train_subset.batch_size * 10)

    model.fit_generator(database.train_subset, epochs=20, validation_data=database.test_subset,
                        callbacks=[tensorboard])

    model.save_weights(os.path.join(log_dir, "weights.h5"))


def main():
    mode = "conv_block"
    model = get_model(mode)

    database = get_ucsd_database(True, False)
    train_subset = database.train_subset
    test_subset = database.test_subset

    model.compile(optimizer=Adam(lr=1e-3), loss="mse", metrics=["mse"])

    # model.load_weights("../logs/tests/conv3d_rec_pred/{}/1551190470/weights.h5".format(mode))

    train_model(model, database, mode)
    visualize_model_errors(model, test_subset)
    visualize_model_errors(model, train_subset)
    evaluate_model_anomaly_detection_on_subset(model, test_subset, 500, 8)


if __name__ == "__main__":
    main()
