import tensorflow as tf
import numpy as np
import csv
import os
import argparse
from shutil import copyfile
from scipy.stats import norm
from scipy.signal import convolve as conv1d
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple, List

from main import run_protocol
from datasets.data_readers import PacketReader
from protocols.Protocol import get_dataset_folder
from protocols.packet_protocols import KitsuneProtocol


class KitsuneTest(object):
    def __init__(self,
                 dataset_name: str,
                 log_dir: str,
                 initial_epoch: int,
                 ):
        dataset_name = KitsuneProtocol.get_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.dataset_folder = get_dataset_folder(dataset_name)
        self.initial_epoch = initial_epoch

        self.protocol = KitsuneProtocol(base_log_dir=log_dir, epoch=self.initial_epoch,
                                        kitsune_dataset=dataset_name)
        self.log_dir = self.protocol.make_log_dir("anomaly_detection")
        self.result_file = None
        self.protocol.load_weights(self.initial_epoch)

    @property
    def model(self):
        return self.protocol.model

    @property
    def train_samples_count(self) -> int:
        return self.protocol.dataset_loader.train_subset.size

    @property
    def benign_samples_count(self) -> int:
        return 1000000 if not self.is_mirai else 100000

    @property
    def is_mirai(self) -> bool:
        return self.dataset_name == "Mirai Botnet"

    @property
    def result_file_header(self) -> str:
        header = "==========" * 10 + "\n"
        header += "|| >>>>>> Results for dataset `{}` using model `{}` at epoch {} <<<<<< ||\n". \
            format(self.dataset_name, self.protocol.model_architecture, self.initial_epoch)
        return header

    def get_labels_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("labels.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_packets_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("dataset.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_labels(self) -> np.ndarray:
        labels_filepath = self.get_labels_csv_filepath()
        with open(labels_filepath, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header
            labels = [row[-1] == "1" for row in csv_reader]
            labels = np.asarray(labels).astype(np.float32)
        return labels

    @staticmethod
    def plot_labels(labels: np.ndarray):
        import matplotlib.pyplot as plt
        start = -1

        for i in range(len(labels)):
            if start == -1:
                if labels[i]:
                    start = i
            else:
                if not labels[i]:
                    plt.gca().axvspan(start, i, alpha=0.25, color="red", linewidth=0)
                    start = -1

        if start != -1:
            plt.gca().axvspan(start, len(labels) - 1, alpha=0.25, color="red", linewidth=0)
        plt.show()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("--- Getting labels ---")
        labels = self.get_labels()

        print("--- Getting packets ---")
        packets_file = self.get_packets_csv_filepath()
        packets_reader = PacketReader(packets_file, max_frame_count=labels.shape[0], discard_first_column=self.is_mirai)
        packets = np.stack([packet for packet in packets_reader], axis=0)

        train_packets = packets[:self.train_samples_count]

        train_min = train_packets.min(axis=0, keepdims=True)
        train_max = train_packets.max(axis=0, keepdims=True)
        train_range = train_max - train_min
        if np.any(train_range == 0):
            train_range = np.where(train_range == 0, np.ones_like(train_min), train_range)

        test_packets = packets[self.benign_samples_count:]
        test_labels = labels[self.benign_samples_count:]

        train_packets = (train_packets - train_min) / train_range
        test_packets = (test_packets - train_min) / train_range

        if self.benign_samples_count > self.train_samples_count:
            benign_packets = packets[self.train_samples_count:self.benign_samples_count]
            benign_packets = (benign_packets - train_min) / train_range
        else:
            benign_packets = None

        return train_packets, benign_packets, test_packets, test_labels

    @tf.function
    def compute_anomaly_score(self, inputs: tf.Tensor):
        outputs = self.model.autoencode(inputs)
        error = tf.abs(inputs - outputs)
        anomaly_score = tf.reduce_mean(error, axis=[1, 2])
        return anomaly_score

    @tf.function
    def compute_anomaly_scores(self, inputs: tf.Tensor):
        batch_size = 256
        samples_length = self.protocol.input_length
        available_samples_count = inputs.shape[0] - samples_length + 1
        loop_iterations_count = available_samples_count // batch_size
        anomaly_scores_init = tf.TensorArray(dtype=tf.float32, size=loop_iterations_count, element_shape=[batch_size])
        loop_vars = (tf.constant(0, dtype=tf.int32), anomaly_scores_init)

        indices = tf.range(samples_length, dtype=tf.int32)
        indices = tf.tile(tf.expand_dims(indices, axis=0), [batch_size, 1])
        offset = tf.range(batch_size, dtype=tf.int32)
        offset = tf.expand_dims(offset, axis=1)
        indices += offset

        def loop_cond(i, _):
            return i < loop_iterations_count

        def loop_body(i, array: tf.TensorArray):
            start = i * batch_size
            x = inputs[start:start + samples_length + batch_size]
            x = tf.gather(x, indices)
            anomaly_score = self.compute_anomaly_score(x)
            array = array.write(i, anomaly_score)
            return i + 1, array

        _, anomaly_scores_array = tf.while_loop(cond=loop_cond,
                                                body=loop_body,
                                                loop_vars=loop_vars)
        anomaly_scores = anomaly_scores_array.stack()
        anomaly_scores = tf.reshape(anomaly_scores, [loop_iterations_count * batch_size])

        remaining_samples_count = available_samples_count - loop_iterations_count * batch_size
        remaining_samples = inputs[-(remaining_samples_count + samples_length):]
        remaining_indices = indices[:remaining_samples_count]
        remaining_samples = tf.gather(remaining_samples, remaining_indices)
        remaining_anomaly_score = self.compute_anomaly_score(remaining_samples)

        anomaly_scores = tf.concat([anomaly_scores, remaining_anomaly_score], axis=0)
        return anomaly_scores

    def get_temporal_labels(self, labels: np.ndarray) -> np.ndarray:
        samples_length = self.protocol.input_length
        kernel = np.ones([samples_length], dtype=np.float32)
        temporal_labels = (conv1d(labels, kernel, mode="valid") >= 1.0).astype(np.float32)
        return temporal_labels

    @staticmethod
    def get_detection_scores(labels, scores):
        predictions = safe_normalize(scores)
        results = {}

        # region ROC / EER
        roc = tf.metrics.AUC(curve="ROC", num_thresholds=100)
        roc.update_state(labels, predictions)

        # region ROC
        results["ROC"] = roc.result()
        # endregion

        # region EER
        tp = roc.true_positives.numpy()
        fp = roc.false_positives.numpy()
        tpr = (tp / tp.max()).astype(np.float64)
        fpr = (fp / fp.max()).astype(np.float64)
        # noinspection PyTypeChecker
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        results["EER"] = eer
        # endregion

        # endregion

        # region PR
        pr = tf.metrics.AUC(curve="PR", num_thresholds=100)
        pr.update_state(labels, predictions)
        results["PR"] = pr.result()
        # endregion

        # region Precision
        thresholds = list(np.arange(0.01, 1.0, 1.0 / 200.0, dtype=np.float32))
        precision = tf.metrics.Precision(thresholds=thresholds)
        recall = tf.metrics.Recall(thresholds=thresholds)

        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)

        recall_result = recall.result().numpy()
        precision_result = precision.result().numpy()
        average_precision = -np.sum(np.diff(recall_result) * precision_result[:-1])

        results["Precision"] = average_precision
        # endregion

        return results

    def run_test(self):
        train_packets, benign_packets, test_packets, test_labels = self.get_data()

        print("--- Computing anomaly scores ---")
        if benign_packets is not None:
            print("--- Train samples ---")
            train_packets = tf.constant(train_packets, dtype=tf.float32)
            train_scores = self.compute_anomaly_scores(train_packets)
            train_log_scores = tf.math.log(train_scores).numpy()
            train_log_mean = train_log_scores.mean()
            train_log_stddev = train_log_scores.std()

            print("--- Benign samples ---")
            benign_packets = tf.constant(benign_packets, dtype=tf.float32)
            benign_scores = self.compute_anomaly_scores(benign_packets)
            benign_log_scores = tf.math.log(benign_scores).numpy()
            benign_log_mean = benign_log_scores.mean()
            benign_log_stddev = benign_log_scores.std()
        else:
            train_scores = train_log_mean = train_log_stddev = None
            benign_scores = benign_log_mean = benign_log_stddev = None

        print("--- Test samples ---")
        test_packets = tf.constant(test_packets, dtype=tf.float32)
        test_scores = self.compute_anomaly_scores(test_packets)
        test_log_error = tf.math.log(test_scores).numpy()
        test_log_mean = test_log_error.mean()
        test_log_stddev = test_log_error.std()
        # test_clipped_error = tf.clip_by_value(test_scores, 0.0, 1.0).numpy()

        print("==========" * 10)
        if benign_packets is not None:
            self.print("Train - Length : {}".format(train_scores.shape[0]))
            self.print("Train - Log Mean : {} | Log Stddev : {}".format(train_log_mean, train_log_stddev))
            self.print("Benign - Length : {}".format(benign_scores.shape[0]))
            self.print("Benign - Log Mean : {} | Log Stddev : {}".format(benign_log_mean, benign_log_stddev))
        self.print("Test - Length : {}".format(test_scores.shape[0]))
        self.print("Test - Log Mean : {} | Log Stddev : {}".format(test_log_mean, test_log_stddev))
        print("==========" * 10)

        metrics = {
            "Log error": test_log_error,
        }

        if benign_packets is not None:
            print("--- Computing log probabilities ---")
            test_log_prob = norm.logsf(test_log_error, benign_log_mean, benign_log_stddev)
            metrics["Log probability"] = test_log_prob

        print("--- Computing temporal labels ---")
        labels = self.get_temporal_labels(test_labels)

        # print("--- Saving data ---")
        # np.save(os.path.join(self.log_dir, "train_log_scores.npy"), train_log_scores)
        # np.save(os.path.join(self.log_dir, "benign_log_scores.npy"), benign_log_scores)
        # np.save(os.path.join(self.log_dir, "test_log_scores.npy"), test_log_error)
        # np.save(os.path.join(self.log_dir, "log_prob.npy"), test_log_prob)
        # np.save(os.path.join(self.log_dir, "base_labels.npy"), test_labels)
        # np.save(os.path.join(self.log_dir, "labels.npy"), labels)

        self.print("==========" * 10)
        self.print("--- Anomaly detection scores ---")
        for metric_name, scores in metrics.items():
            self.evaluate_scores(labels, scores, metric_name)
        self.print("==========" * 10)

    def evaluate_scores(self, labels: np.ndarray, scores: np.ndarray, name: str):
        self.print("----- {} -----".format(name))
        scores = self.get_detection_scores(labels, scores)
        for metric_name, metric_value in scores.items():
            self.print("{} : {}".format(metric_name, metric_value))

    def print(self, *values):
        if self.result_file is None:
            dataset_name = self.dataset_name.lower().replace(" ", "_")
            result_filepath = os.path.join(self.log_dir, "{}_results.txt".format(dataset_name))
            self.result_file = open(result_filepath, "w")
            print(self.result_file_header, self.result_file, flush=False)

        print(*values, file=self.result_file, flush=True)
        print(*values)


def safe_normalize(array: np.ndarray) -> np.ndarray:
    # noinspection PyArgumentList
    array_min = array.min(axis=0, keepdims=True)
    # noinspection PyArgumentList
    array_max = array.max(axis=0, keepdims=True)

    array_range = array_max - array_min
    if np.any(array_range == 0):
        array_range = np.where(array_range == 0, np.ones_like(array_min), array_range)

    return (array - array_min) / array_range


def copy_weights(datasets: List[str], model: str, epoch: int, log_dir: str):
    print("======" * 10)
    print("--- Copying latest weights for testing ---")

    for dataset in datasets:
        dataset_folder = os.path.join(log_dir, "packet", dataset)
        train_folder = os.path.join(dataset_folder, "train", model)
        trainings = sorted(os.listdir(train_folder))
        last_training = trainings[-1]
        weights_folder = os.path.join(train_folder, last_training)
        weights_id = "weights_{:03d}".format(epoch)
        weights_filenames = [filename for filename in os.listdir(weights_folder) if filename.startswith(weights_id)]
        if len(weights_filenames) == 0:
            raise ValueError("Could not find weight at epoch {} for model {} in {}".format(epoch, model,
                                                                                           weights_folder))
        else:
            print("> Copying {} `{}` from `{}` to `{}`.".format(len(weights_filenames), weights_id, weights_folder,
                                                                dataset_folder))

        for filename in weights_filenames:
            origin = os.path.join(weights_folder, filename)
            target = os.path.join(dataset_folder, filename)
            copyfile(origin, target)

    print("======" * 10)


def run_train_protocol(dataset: str, epoch: int, log_dir: str):
    run_protocol(dataset, mode="train", epoch=epoch, log_dir=log_dir)


def run_test_protocol(dataset: str, epoch: int, log_dir: str):
    test = KitsuneTest(dataset_name=dataset,
                       log_dir=log_dir,
                       initial_epoch=epoch)
    test.run_test()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", required=True)
    arg_parser.add_argument("--epoch", required=True)
    arg_parser.add_argument("--log_dir", default="../logs/AEA")
    arg_parser.add_argument("--mode", default="train_test")

    args = arg_parser.parse_args()
    epoch = int(args.epoch)
    model: str = args.model
    log_dir: str = args.log_dir
    mode: str = args.mode

    datasets = ["Active Wiretap", "ARP MitM", "Fuzzing",
                "Mirai Botnet", "OS Scan", "SSDP Flood",
                "SSL Renegotiation", "SYN DoS", "Video Injection"]

    if "train" in mode:
        for dataset in datasets:
            run_train_protocol(dataset, epoch=0, log_dir=log_dir)

    if "test" in mode:
        copy_weights(datasets, model=model, epoch=epoch, log_dir=log_dir)
        for dataset in datasets:
            run_test_protocol(dataset, epoch=epoch, log_dir=log_dir)


if __name__ == "__main__":
    main()
