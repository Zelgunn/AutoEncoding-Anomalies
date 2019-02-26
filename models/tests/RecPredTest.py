import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, Deconv3D, BatchNormalization, LeakyReLU, MaxPooling3D, concatenate, UpSampling3D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K
import os
from time import time
import cv2
import numpy as np
from tqdm import tqdm

from layers import ResBlock3D, DenseBlock3D, ResBlock3DTranspose
from datasets import UCSDDatabase
from data_preprocessors import BrightnessShifter, DropoutNoiser


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

    layer = ResBlock3D(kernel_size=3, filters=32, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBlock3D(kernel_size=3, filters=48, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBlock3D(kernel_size=3, filters=64, kernel_initializer="he_normal")(layer)
    layer = MaxPooling3D(pool_size=2, strides=2)(layer)

    layer = ResBlock3D(kernel_size=3, filters=64, kernel_initializer="he_normal")(layer)

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

    layer = ResBlock3DTranspose(kernel_size=3, filters=48, kernel_initializer="he_normal", strides=2)(layer)
    layer = ResBlock3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", strides=2)(layer)
    layer = ResBlock3DTranspose(kernel_size=3, filters=32, kernel_initializer="he_normal", strides=2)(layer)

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

    # layer = DenseBlock3D(kernel_size=3, growth_rate=12, depth=4, output_filters=32)(layer)
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
    return model


def get_ucsd_database(shift_brightness, dropout_noise):
    train_preprocessors = []
    if shift_brightness:
        train_preprocessors.append(BrightnessShifter(0.25, 0.25, 0.0, 0.0))
    if dropout_noise:
        train_preprocessors.append(DropoutNoiser(0.2, 0.0))

    ucsd_database = UCSDDatabase(database_path="../datasets/ucsd/ped2",
                                 input_sequence_length=None,
                                 output_sequence_length=None,
                                 train_preprocessors=train_preprocessors)
    ucsd_database.load()

    ucsd_database = ucsd_database.resized(image_size=(128, 128), input_sequence_length=16, output_sequence_length=32)
    ucsd_database.normalize(0.0, 1.0)

    ucsd_database.train_dataset.epoch_length = 250
    ucsd_database.train_dataset.batch_size = 8
    ucsd_database.test_dataset.epoch_length = 25
    ucsd_database.test_dataset.batch_size = 2

    return ucsd_database


def visualize(model, dataset):
    key = 13
    escape_key = 27
    seed = 0
    i = 0

    y_pred, y_true, tmp = None, None, None

    while key != escape_key:
        if key != -1:
            seed += 1
            x, y_true = dataset.get_batch(seed=seed)
            y_pred = model.predict(x)
            tmp = np.abs(y_pred - y_true).max()
            i = 0

        y_pred_resized = cv2.resize(y_pred[0][i], dsize=(512, 512))
        y_true_resized = cv2.resize(y_true[0][i], dsize=(512, 512))
        delta = np.abs(y_pred_resized - y_true_resized) / tmp
        # delta = (delta - delta.min()) / (delta.max() - delta.min())
        # delta = np.square(delta)

        composite = np.zeros(shape=[512, 512], dtype=np.float32)
        composite = np.repeat(composite[:, :, np.newaxis], 3, axis=2)

        composite[..., 0] = (1.0 - delta) * 90
        composite[..., 1] = delta
        composite[..., 2] = y_true_resized
        composite = cv2.cvtColor(composite, cv2.COLOR_HSV2BGR)

        cv2.imshow("y_true", y_true_resized)
        cv2.imshow("y_pred", y_pred_resized)
        cv2.imshow("delta", delta)
        cv2.imshow("composite", composite)
        key = cv2.waitKey(30)

        i = (i + 1) % 32


def train_model(model, database, mode):
    log_dir = "../logs/tests/conv3d_rec_pred/{}/{}".format(mode, int(time()))
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq=database.train_dataset.batch_size * 10)

    model.fit_generator(database.train_dataset, epochs=20, validation_data=database.test_dataset,
                        callbacks=[tensorboard])

    model.save_weights(os.path.join(log_dir, "weights.h5"))


def test_anomaly_detection(model, dataset):
    dataset.epoch_length = 500
    dataset.batch_size = 8

    input_layer = model.input
    pred_output = model.output
    true_output = tf.placeholder(dtype=tf.float32, shape=model.output_shape)
    error_op = tf.square(pred_output - true_output)
    error_op = tf.reduce_sum(error_op, axis=[1, 2, 3, 4])

    session = K.get_session()

    errors = np.zeros(shape=[dataset.epoch_length * dataset.batch_size])
    labels = np.zeros(shape=[dataset.epoch_length * dataset.batch_size], dtype=np.bool)
    for i in tqdm(range(dataset.epoch_length), desc="Computing errors..."):
        images, step_labels, _ = dataset.sample(return_labels=True)
        x, y_true = dataset.divide_batch_io(images)
        x, y_true = dataset.apply_preprocess(x, y_true)

        step_error = session.run(error_op, feed_dict={input_layer: x, true_output: y_true})
        indices = np.arange(i * dataset.batch_size, (i + 1) * dataset.batch_size)

        errors[indices] = step_error
        labels[indices] = np.any(step_labels, axis=1)

    errors = (errors - errors.min()) / (errors.max() - errors.min())
    print(np.mean(labels.astype(np.float32)))

    labels = tf.constant(labels)
    errors = tf.constant(errors)
    roc_ops = tf.metrics.auc(labels, errors, summation_method="careful_interpolation")
    pr_ops = tf.metrics.auc(labels, errors, curve="PR", summation_method="careful_interpolation")
    auc_ops = roc_ops + pr_ops

    session.run(tf.local_variables_initializer())
    _, roc, _, pr = session.run(auc_ops)
    print("ROC : {} | PR : {}".format(roc, pr))


def main():
    mode = "residual_block"
    model = get_model(mode)

    database = get_ucsd_database(True, False)
    train_dataset = database.train_dataset
    test_dataset = database.test_dataset

    model.compile(optimizer=Adam(lr=1e-4), loss="mse", metrics=["mse", "mae"])

    model.load_weights("../logs/tests/conv3d_rec_pred/{}/1551168759/weights.h5".format(mode))

    # train_model(model, database, mode)
    visualize(model, train_dataset)
    test_anomaly_detection(model, train_dataset)


if __name__ == "__main__":
    main()
