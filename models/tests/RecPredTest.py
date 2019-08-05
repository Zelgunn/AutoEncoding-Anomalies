from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv3D, Conv3DTranspose
from tensorflow.python.keras.layers import BatchNormalization, LeakyReLU, concatenate
from tensorflow.python.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
import os
from time import time

from layers import ResBasicBlock3D, DenseBlock3D, ResBasicBlock3DTranspose
from datasets import DatasetConfig, DatasetLoader


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

    layer = Conv3DTranspose(kernel_size=3, filters=48, kernel_initializer="he_normal", padding="same", strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    layer = Conv3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", padding="same", strides=2)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)

    layer = Conv3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", padding="same", strides=2)(layer)
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


def get_ucsd_dataset():
    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           output_range=(0.0, 1.0))

    dataset = DatasetLoader(config)
    return dataset


def get_subway_dataset():
    config = DatasetConfig(tfrecords_config_folder="../datasets/subway/exit",
                           output_range=(0.0, 1.0))

    dataset = DatasetLoader(config)
    return dataset


def train_model(model, dataset, mode):
    log_dir = "../logs/tests/conv3d_rec_pred/{}/{}".format(mode, int(time()))
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq=dataset.train_subset.batch_size * 10)

    model.fit_generator(dataset.train_subset, epochs=20, validation_data=dataset.test_subset,
                        callbacks=[tensorboard])

    model.save_weights(os.path.join(log_dir, "weights.h5"))


def main():
    mode = "conv_block"
    model = get_model(mode)

    dataset = get_ucsd_dataset()
    # train_subset: SubsetLoader = dataset.train_subset
    # test_subset: SubsetLoader = dataset.test_subset

    model.compile(optimizer=Adam(lr=1e-3), loss="mse", metrics=["mse"])

    # model.load_weights("../logs/tests/conv3d_rec_pred/{}/1551190470/weights.h5".format(mode))

    train_model(model, dataset, mode)
    # visualize_model_errors(model, test_subset)
    # visualize_model_errors(model, train_subset)
    # evaluate_model_anomaly_detection_on_subset(model, test_subset, 500, 8)


if __name__ == "__main__":
    main()
