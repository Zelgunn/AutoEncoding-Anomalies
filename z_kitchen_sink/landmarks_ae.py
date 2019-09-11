import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate, Flatten, Reshape
from tensorflow.python.keras.layers import Conv1D, AveragePooling1D, UpSampling1D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import os
from time import time

from callbacks import AUCCallback, LandmarksVideoCallback
from modalities import ModalityLoadInfo, Landmarks, Pattern
from anomaly_detection import RawPredictionsModel, AnomalyDetector
from z_kitchen_sink.utils import get_landmarks_datasets, get_temporal_loss_weights


def make_landmarks_encoder(input_layer, sequence_length):
    common_args = {"kernel_size": 3,
                   "padding": "same",
                   "activation": "relu",
                   "use_bias": False}

    layer = input_layer
    layer = Reshape([sequence_length, 68 * 2])(layer)

    layer = Conv1D(filters=64, **common_args)(layer)
    layer = Conv1D(filters=64, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = AveragePooling1D(pool_size=2, strides=2)(layer)
    layer = Flatten()(layer)
    layer = Dense(units=64, activation="relu")(layer)

    return layer


def make_landmarks_decoder(input_layer, sequence_length):
    common_args = {"kernel_size": 3,
                   "padding": "same",
                   "activation": "relu",
                   "use_bias": False}

    layer = input_layer

    layer = Dense(units=8 * 16, activation="relu")(layer)
    layer = Reshape([8, 16])(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = Conv1D(filters=16, **common_args)(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = Conv1D(filters=32, **common_args)(layer)
    layer = UpSampling1D(size=2)(layer)
    layer = Conv1D(filters=64, **common_args)(layer)
    layer = Conv1D(filters=68 * 2, kernel_size=3, padding="same", activation=None, use_bias=False)(layer)

    layer = Reshape([sequence_length, 68, 2])(layer)

    return layer


def make_landmarks_autoencoder(sequence_length: int, add_predictor: bool):
    input_shape = (sequence_length, 68, 2)

    input_layer = Input(input_shape)

    encoded = make_landmarks_encoder(input_layer, sequence_length)
    decoded = make_landmarks_decoder(encoded, sequence_length)
    if add_predictor:
        predicted = make_landmarks_decoder(encoded, sequence_length)
        decoded = Concatenate(axis=1)([decoded, predicted])

    output_layer = decoded

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="landmarks_autoencoder")

    # loss = get_autoencoder_loss(sequence_length, axis=(2, 3)) if add_predictor else "mse"

    def landmarks_loss(y_true, y_pred):
        indexes = [36, 39, 42, 45] + list(range(48, 68))

        mask = []
        for i in range(68):
            mask.append(1.0 if i in indexes else 1e-1)

        mask = tf.constant(mask, shape=[1, 1, 68, 1])
        loss = tf.square(y_true - y_pred) * mask
        loss = tf.reduce_mean(loss, axis=[2, 3])
        loss *= get_temporal_loss_weights(sequence_length, sequence_length)
        loss = tf.reduce_mean(loss, axis=1)
        return loss

    optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=0.0)
    autoencoder.compile(optimizer, loss=landmarks_loss)

    return autoencoder


def train_landmarks_autoencoder():
    add_predictor = True
    input_length = 64
    output_length = input_length * 2 if add_predictor else input_length
    batch_size = 16

    landmarks_autoencoder = make_landmarks_autoencoder(sequence_length=input_length,
                                                       add_predictor=add_predictor)
    # landmarks_autoencoder.load_weights("../logs/tests/kitchen_sink/mfcc_only/weights_020.hdf5")

    dataset_loader, train_subset, test_subset = get_landmarks_datasets()
    pattern = Pattern(
        ModalityLoadInfo(Landmarks, input_length, (input_length, 136)),
        ModalityLoadInfo(Landmarks, output_length, (output_length, 136))
    )

    train_dataset = train_subset.make_tf_dataset(pattern)
    train_dataset = train_dataset.batch(batch_size).prefetch(-1)

    test_dataset = test_subset.make_tf_dataset(pattern)
    test_dataset = test_dataset.batch(batch_size)

    # region Callbacks
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/landmarks"
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    raw_predictions_model = RawPredictionsModel(landmarks_autoencoder, output_length)
    auc_callback = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=test_subset, pattern=Pattern(*pattern, "labels"))

    train_landmarks_callback = LandmarksVideoCallback(train_subset, landmarks_autoencoder, tensorboard, pattern=pattern)
    test_landmarks_callback = LandmarksVideoCallback(test_subset, landmarks_autoencoder, tensorboard, pattern=pattern)

    model_checkpoint = ModelCheckpoint(os.path.join(base_log_dir, "weights.{epoch:03d}.hdf5", ))
    callbacks = [tensorboard,
                 train_landmarks_callback,
                 test_landmarks_callback,
                 auc_callback,
                 model_checkpoint]

    summary_filename = os.path.join(log_dir, "{}_summary.txt".format(landmarks_autoencoder.name))
    with open(summary_filename, "w") as file:
        landmarks_autoencoder.summary(print_fn=lambda summary: file.write(summary + '\n'))

    # endregion

    landmarks_autoencoder.fit(train_dataset, epochs=10, steps_per_epoch=10000,
                              validation_data=test_dataset, validation_steps=200,
                              callbacks=callbacks)
    # landmarks_autoencoder.load_weights(os.path.join(base_log_dir, "weights_006.hdf5"))

    anomaly_detector = AnomalyDetector(autoencoder=landmarks_autoencoder,
                                       output_length=output_length)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=pattern,
                                          log_dir=log_dir,
                                          stride=1)
