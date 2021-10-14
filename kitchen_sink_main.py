import tensorflow as tf
import numpy as np
import os
from scipy.signal import convolve as conv1d
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from typing import Union

from protocols.packet_protocols import KitsuneProtocol


def get_base_data():
    dataset_folder = r"E:\Users\Degva\Documents\_PhD\Tensorflow\datasets\kitsune\Active Wiretap"
    packets = np.load(os.path.join(dataset_folder, "dataset.npy"))
    train_packets = packets[:1000000]
    test_packets = packets[1000000:]

    labels = np.load(os.path.join(dataset_folder, "labels.npy"))
    labels = labels[1000000:]

    return train_packets, test_packets, labels


@tf.function
def interpolation_error(latent_codes: tf.Tensor) -> tf.Tensor:
    step_count = latent_codes.shape[1]
    first_latent_code = latent_codes[:, 0]
    last_latent_code = latent_codes[:, -1]

    target_latent_codes = temporal_interpolation(first_latent_code, last_latent_code, step_count)

    latent_codes = latent_codes[:, 1:-1]
    target_latent_codes = target_latent_codes[:, 1:-1]

    error = tf.square(latent_codes - target_latent_codes)
    # error = tf.abs(latent_codes - target_latent_codes)

    return error


@tf.function
def temporal_interpolation(first_latent_code: tf.Tensor,
                           last_latent_code: tf.Tensor,
                           step_count: Union[tf.Tensor, int]
                           ) -> tf.Tensor:
    first_latent_code = tile_latent_code(first_latent_code, step_count, stop_gradient=True)
    last_latent_code = tile_latent_code(last_latent_code, step_count, stop_gradient=True)

    weights_shape = [1, step_count] + [1] * (len(first_latent_code.shape) - 2)
    weights = tf.range(start=0, limit=step_count, dtype=tf.int32)
    weights = tf.cast(weights / (step_count - 1), tf.float32)
    weights = tf.reshape(weights, weights_shape)

    latent_codes = first_latent_code * (1.0 - weights) + last_latent_code * weights
    return latent_codes


@tf.function
def tile_latent_code(latent_code: tf.Tensor,
                     step_count: int,
                     stop_gradient: bool):
    if stop_gradient:
        latent_code = tf.stop_gradient(latent_code)

    tile_multiples = [1, step_count] + [1] * (len(latent_code.shape) - 1)
    latent_code = tf.expand_dims(latent_code, axis=1)
    latent_code = tf.tile(latent_code, tile_multiples)

    return latent_code


def compute_labels(input_length):
    dataset_folder = r"E:\Users\Degva\Documents\_PhD\Tensorflow\datasets\kitsune\Active Wiretap"
    labels = np.load(os.path.join(dataset_folder, "labels.npy"))
    labels = labels[1000000:]

    kernel = np.ones([input_length], dtype=np.float32)
    labels = (conv1d(labels, kernel, mode="valid") >= 1.0).astype(np.float32)

    np.save(r"C:\Users\Degva\Desktop\tmp\work\LTM\labels.npy", labels)
    return labels


def compute_base_predictions():
    protocol = KitsuneProtocol(base_log_dir="../logs/AEA", epoch=128, kitsune_dataset="Active Wiretap")
    protocol.load_weights(128)
    model = protocol.model

    batch_size = 1024

    pattern = protocol.get_network_packet_pattern()
    train_dataset = protocol.dataset_loader.train_subset.make_base_tf_dataset(pattern, 1, None, False, batch_size)
    test_dataset = protocol.dataset_loader.test_subset.make_base_tf_dataset(pattern, 1, None, False, batch_size)

    train_predictions = model.predict(train_dataset, verbose=1)
    test_predictions = model.predict(test_dataset, verbose=1)

    np.save(r"C:\Users\Degva\Desktop\tmp\work\LTM\train_predictions.npy", train_predictions)
    np.save(r"C:\Users\Degva\Desktop\tmp\work\LTM\test_predictions.npy", test_predictions)

    return train_predictions, test_predictions


def compute_initial():
    train_predictions, test_predictions = compute_base_predictions()
    labels = compute_labels(input_length=64)
    return train_predictions, test_predictions, labels


def reload():
    train_predictions = np.load(r"C:\Users\Degva\Desktop\tmp\work\LTM\train_predictions.npy")
    test_predictions = np.load(r"C:\Users\Degva\Desktop\tmp\work\LTM\test_predictions.npy")
    labels = np.load(r"C:\Users\Degva\Desktop\tmp\work\LTM\labels.npy")
    return train_predictions, test_predictions, labels


def random_choice_nd(array: np.ndarray, size: int) -> np.ndarray:
    indices = np.arange(array.shape[0])
    indices = np.random.choice(indices, size)
    return array[indices]


def normalize_sklearn_predictions(predictions):
    predictions = - predictions
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    return predictions


def run_isolation_forest(train_predictions: np.ndarray, test_predictions: np.ndarray) -> np.ndarray:
    isolation_forest = IsolationForest(max_samples=0.8, n_jobs=-1)
    print("IF - Fitting...")
    isolation_forest.fit(train_predictions)

    print("IF - Predicting...")
    predictions = isolation_forest.score_samples(test_predictions)
    predictions = normalize_sklearn_predictions(predictions)
    return predictions


def run_svm(train_predictions: np.ndarray, test_predictions: np.ndarray) -> np.ndarray:
    svm = OneClassSVM(kernel="rbf", nu=0.03, cache_size=8000)
    train_predictions = random_choice_nd(train_predictions, size=10000)

    print("SVM - Fitting...")
    svm.fit(train_predictions)

    print("SVM - Predicting...")
    predictions = svm.score_samples(test_predictions)
    predictions = normalize_sklearn_predictions(predictions)
    return predictions


def main():
    # train_packets, test_packets, base_labels = get_base_data()
    # svm_predictions = run_svm(train_packets, test_packets)
    # svm_result = roc_auc_score(base_labels.astype("bool"), svm_predictions)
    # print("[Base Data] SVM Result :", round(svm_result, 3))

    # forest_predictions = run_isolation_forest(train_packets, test_packets)
    # forest_result = roc_auc_score(base_labels.astype("bool"), forest_predictions)
    # print("[Base Data] Isolation Forest Result :", round(forest_result, 3))

    train_predictions, test_predictions, labels = compute_initial()
    # train_predictions, test_predictions, labels = reload()
    # train_predictions = interpolation_error(tf.convert_to_tensor(train_predictions)).numpy()
    # test_predictions = interpolation_error(tf.convert_to_tensor(test_predictions)).numpy()
    train_predictions = np.reshape(train_predictions, [train_predictions.shape[0], -1])
    test_predictions = np.reshape(test_predictions, [test_predictions.shape[0], -1])

    # svm_predictions = run_svm(train_predictions, test_predictions)
    # np.save(r"C:\Users\Degva\Desktop\tmp\work\LTM\svm_predictions.npy", svm_predictions)
    # svm_result = roc_auc_score(labels.astype("bool"), svm_predictions)
    # print("[LTM Data] SVM Result :", round(svm_result, 3))

    forest_predictions = run_isolation_forest(train_predictions, test_predictions)
    np.save(r"C:\Users\Degva\Desktop\tmp\work\LTM\forest_predictions.npy", forest_predictions)
    forest_result = roc_auc_score(labels.astype("bool"), forest_predictions)
    print("[LTM Data] Isolation Forest Result :", round(forest_result, 3))


if __name__ == "__main__":
    main()
