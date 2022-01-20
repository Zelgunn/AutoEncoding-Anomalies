import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as keras_backend
import numpy as np
import csv
import os
import json
import argparse
from time import time
from shutil import copyfile
from scipy.stats import pearsonr
from scipy.signal import convolve as conv1d
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Optional, Callable

from main import run_protocol
from datasets.data_readers import KitsunePacketReader
from protocols.Protocol import get_dataset_folder
from protocols.packet_protocols import KitsuneProtocol
from misc_utils.general import int_ceil
from misc_utils.numpy_utils import safe_normalize, safe_standardize
from custom_tf_models import VIAE, CnC, LED, LTM


class KitsuneTest(object):
    def __init__(self,
                 dataset_name: str,
                 log_dir: str,
                 initial_epoch: int,
                 num_thresholds=1000,
                 config: Dict = None,
                 load_weights=True,
                 score_mode="auto"
                 ):
        dataset_name = KitsuneProtocol.get_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.dataset_folder = get_dataset_folder(dataset_name)
        self.initial_epoch = initial_epoch
        self.num_thresholds = num_thresholds

        self.protocol = KitsuneProtocol(base_log_dir=log_dir, epoch=self.initial_epoch,
                                        kitsune_dataset=dataset_name, config=config)
        self._log_dir = None
        self.result_file = None
        if load_weights:
            self.protocol.load_weights(self.initial_epoch)

        self.roc = tf.metrics.AUC(curve="ROC", num_thresholds=self.num_thresholds)
        self.pr = tf.metrics.AUC(curve="PR", num_thresholds=self.num_thresholds)

        if not isinstance(self.model, (CnC, LED, LTM)):
            score_mode = "reconstruction"
        elif score_mode == "auto":
            score_mode = "combined"
        elif score_mode not in ["reconstruction", "energy", "combined"]:
            raise ValueError("`score_mode` ({}) not in [\"reconstruction\", \"energy\", \"combined\"] "
                             .format(score_mode))

        self.score_mode = score_mode

    @property
    def log_dir(self) -> str:
        if self._log_dir is None:
            self._log_dir = self.protocol.make_log_dir("anomaly_detection")
        return self._log_dir

    @property
    def model(self):
        return self.protocol.model

    @property
    def sample_length(self) -> int:
        return self.protocol.input_length

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

    def get_labels_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "labels.npy")

    def get_packets_csv_filepath(self) -> str:
        files = os.listdir(self.dataset_folder)
        filename = [name for name in files if name.endswith("dataset.csv")][0]
        filepath = os.path.join(self.dataset_folder, filename)
        return filepath

    def get_packets_numpy_filepath(self) -> str:
        return os.path.join(self.dataset_folder, "dataset.npy")

    def get_labels(self) -> np.ndarray:
        if not os.path.exists(self.get_labels_numpy_filepath()):
            labels_filepath = self.get_labels_csv_filepath()
            with open(labels_filepath, "r") as file:
                csv_reader = csv.reader(file)
                if not self.is_mirai:
                    next(csv_reader)  # skip header
                labels = [row[-1] == "1" for row in csv_reader]
                labels = np.asarray(labels).astype(np.float32)
            np.save(self.get_labels_numpy_filepath(), labels)
        else:
            labels = np.load(self.get_labels_numpy_filepath())
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

    def get_packets(self, max_frame_count: int = None) -> np.ndarray:
        if not os.path.exists(self.get_packets_numpy_filepath()):
            packets_file = self.get_packets_csv_filepath()
            packets_reader = KitsunePacketReader(packets_file, max_frame_count=max_frame_count, is_mirai=self.is_mirai)
            packets = np.stack([packet for packet in packets_reader], axis=0)
            np.save(self.get_packets_numpy_filepath(), packets)
        else:
            packets = np.load(self.get_packets_numpy_filepath())

        packets = self.preprocess_data(packets)

        return packets

    def preprocess_data(self, packets):
        if self.protocol.output_activation == "linear":
            packets = safe_standardize(packets)
        else:
            train_packets = packets[:self.benign_samples_count]

            train_min = train_packets.min(axis=0, keepdims=True)
            train_max = train_packets.max(axis=0, keepdims=True)
            train_range = train_max - train_min
            if np.any(train_range == 0):
                train_range = np.where(train_range == 0, np.ones_like(train_min), train_range)

            packets = (packets - train_min) / train_range

        if self.protocol.output_activation == "tanh":
            packets = packets * 2.0 - 1.0

        return packets

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("--- Getting labels ---")
        labels = self.get_labels()

        print("--- Getting packets ---")
        packets = self.get_packets(max_frame_count=labels.shape[0])

        train_packets = packets[:self.train_samples_count]
        test_packets = packets[self.benign_samples_count:]
        test_labels = labels[self.benign_samples_count:]

        if self.benign_samples_count > self.train_samples_count:
            benign_packets = packets[self.train_samples_count:self.benign_samples_count]
        else:
            benign_packets = None

        return train_packets, benign_packets, test_packets, test_labels

    def compute_anomaly_score(self, inputs: tf.Tensor):
        if isinstance(self.model, VIAE):
            anomaly_score = self.compute_epsilon_anomaly_score(inputs)
        elif isinstance(self.model, CnC):
            anomaly_score = self.model.mean_relevance_energy(inputs)
        elif isinstance(self.model, LED):
            anomaly_score = self.model.compute_description_energy(inputs)
        else:
            anomaly_score = self.compute_base_anomaly_score(inputs)
        return anomaly_score

    def compute_base_anomaly_score(self, inputs: tf.Tensor):
        outputs = self.model.autoencode(inputs)
        error = tf.abs(inputs - outputs)
        anomaly_score = tf.reduce_mean(error, axis=[1, 2])
        return anomaly_score

    def compute_epsilon_anomaly_score(self, inputs: tf.Tensor):
        outputs = self.model.autoencode_sampling_epsilon(inputs)
        error = tf.abs(inputs - outputs)
        anomaly_score = tf.reduce_mean(error, axis=[1, 2])
        return anomaly_score

    def compute_final_anomaly_scores(self, inputs: tf.Tensor, use_log=True) -> tf.Tensor:
        if self.score_mode == "reconstruction":
            anomaly_scores = self.compute_reconstruction_anomaly_scores(inputs, use_log=use_log)
        elif self.score_mode == "energy":
            anomaly_scores = self.compute_energy_anomaly_scores(inputs, use_log=use_log)
        elif self.score_mode == "combined":
            if isinstance(self.model, (CnC, LED)):
                reconstruction_scores = self.compute_reconstruction_anomaly_scores(inputs, use_log=use_log)
                energy_scores = self.compute_energy_anomaly_scores(inputs, use_log=use_log)
                anomaly_scores = self.tf_normalize(reconstruction_scores) + self.tf_normalize(energy_scores)
            elif isinstance(self.model, LTM):
                interpolation_scores = self.compute_anomaly_scores(inputs, self.sample_length,
                                                                   self.model.compute_interpolation_error, use_log)
                norm_scores = self.compute_anomaly_scores(inputs, self.sample_length,
                                                          self.model.compute_norm_error, use_log)
                anomaly_scores = self.tf_normalize(interpolation_scores) + self.tf_normalize(norm_scores)
            else:
                raise RuntimeError
        else:
            raise RuntimeError

        return anomaly_scores

    def compute_reconstruction_anomaly_scores(self, inputs: tf.Tensor, use_log=True) -> tf.Tensor:
        if isinstance(self.model, VIAE):
            score_function = self.compute_epsilon_anomaly_score
        else:
            score_function = self.compute_base_anomaly_score
        anomaly_scores = self.compute_anomaly_scores(inputs, self.sample_length, score_function, use_log)
        return anomaly_scores

    def compute_energy_anomaly_scores(self, inputs: tf.Tensor, use_log=True) -> tf.Tensor:
        if isinstance(self.model, CnC):
            score_function = self.model.mean_relevance_energy
        elif isinstance(self.model, LED):
            score_function = self.model.compute_description_energy
        elif isinstance(self.model, LTM):
            # score_function = self.model.compute_norm_error
            score_function = self.model.compute_interpolation_error
        else:
            raise RuntimeError
        anomaly_scores = self.compute_anomaly_scores(inputs, self.sample_length, score_function, use_log)
        return anomaly_scores

    @staticmethod
    def compute_anomaly_scores(inputs: tf.Tensor, samples_length: int, score_function: Callable, use_log=True):
        batch_size = 256
        available_samples_count = tf.shape(inputs)[0] - samples_length + 1
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
            anomaly_score = score_function(x)
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
        remaining_anomaly_score = score_function(remaining_samples)

        anomaly_scores = tf.concat([anomaly_scores, remaining_anomaly_score], axis=0)
        if use_log:
            anomaly_scores = KitsuneTest.get_log_scores(anomaly_scores)

        return anomaly_scores

    @staticmethod
    def get_log_scores(anomaly_scores: tf.Tensor):
        def remove_min() -> tf.Tensor:
            min_value = tf.reduce_min(anomaly_scores)
            return anomaly_scores - min_value + 1e-7

        any_below_one = tf.reduce_any(anomaly_scores < tf.constant(1e-7, dtype=tf.float32))
        anomaly_scores = tf.cond(any_below_one, remove_min, lambda: anomaly_scores)
        anomaly_scores = tf.math.log(anomaly_scores)
        return anomaly_scores

    @staticmethod
    def tf_normalize(tensor: tf.Tensor) -> tf.Tensor:
        tensor_min = tf.reduce_min(tensor)
        tensor_max = tf.reduce_max(tensor)
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    def get_temporal_labels(self, labels: np.ndarray) -> np.ndarray:
        samples_length = self.sample_length
        kernel = np.ones([samples_length], dtype=np.float32)
        temporal_labels = (conv1d(labels, kernel, mode="valid") >= 1.0).astype(np.float32)
        return temporal_labels

    def get_detection_scores(self, labels, scores) -> Dict[str, float]:
        results = {}

        predictions = safe_normalize(scores)
        tf_labels = tf.convert_to_tensor(labels)
        tf_predictions = tf.convert_to_tensor(predictions)

        self.update_roc_and_pr(tf_labels, tf_predictions)
        results["ROC"] = self.roc.result().numpy()
        results["PR"] = self.pr.result().numpy()

        def _get_rate(_tensor: tf.Tensor) -> np.ndarray:
            _array = _tensor.numpy()
            return (_array / _array.max()).astype(np.float64)

        tpr = _get_rate(self.roc.true_positives)
        fpr = _get_rate(self.roc.false_positives)
        fnr = _get_rate(self.roc.false_negatives)

        # noinspection PyTypeChecker
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        results["EER"] = eer

        tpr = tpr[1:-1]
        fpr = fpr[1:-1]
        fnr = fnr[1:-1]

        precision = tpr / (tpr + fpr)
        recall = tpr / (tpr + fnr)
        recall = np.concatenate([recall, [0.0]], axis=0)
        average_precision = sum((recall[:-1] - recall[1:]) * precision)

        results["mAP"] = average_precision

        return results

    def update_roc_and_pr(self, labels: tf.Tensor, predictions: tf.Tensor):
        self.roc.reset_states()
        self.pr.reset_states()
        sample_count = predictions.shape[0]
        batch_size = sample_count // self.num_thresholds
        batch_count = int_ceil(sample_count / batch_size)

        for i in range(batch_count):
            start = i * batch_size
            end = (i + 1) * batch_size
            step_labels = labels[start: end]
            step_predictions = predictions[start: end]
            self.roc.update_state(step_labels, step_predictions)
            self.pr.update_state(step_labels, step_predictions)

    def run_test(self, use_log_score: bool) -> Dict[str, float]:
        _, _, test_packets, test_labels = self.get_data()

        test_packets = tf.constant(test_packets, dtype=tf.float32)
        # anomaly_scores = self.compute_anomaly_scores(test_packets, use_log=use_log_score)
        anomaly_scores = self.compute_final_anomaly_scores(test_packets, use_log=use_log_score)
        anomaly_scores = anomaly_scores.numpy()

        labels = self.get_temporal_labels(test_labels)

        self.print("==========" * 10)
        detection_scores = self.get_detection_scores(labels, anomaly_scores)
        for metric_name, metric_value in detection_scores.items():
            self.print("{} : {}".format(metric_name, metric_value))
        self.print("==========" * 10)

        return detection_scores

    def print(self, *values):
        if self.result_file is None:
            dataset_name = self.dataset_name.lower().replace(" ", "_")
            result_filepath = os.path.join(self.log_dir, "{}_results.txt".format(dataset_name))
            self.result_file = open(result_filepath, "w")
            self.print(self.result_file_header)

        print(*values, file=self.result_file, flush=True)
        print(*values)

    def compute_generalisation_score(self,
                                     n_partitions=5,
                                     n_percentiles=5,
                                     method="kld",
                                     anomaly_scores: tf.Tensor = None
                                     ):
        if anomaly_scores is None:
            packets = self.get_packets()
            packets = packets[:self.train_samples_count]
            packets = tf.convert_to_tensor(packets, dtype=tf.float32)
            anomaly_scores = self.compute_final_anomaly_scores(packets, use_log=True)

        if n_partitions > 0:
            anomaly_scores = tf.random.shuffle(anomaly_scores)

            partition_size = tf.cast(tf.math.ceil(tf.shape(anomaly_scores)[0] / n_partitions), tf.int32)
            partitions = []
            for i in range(n_partitions):
                partition = anomaly_scores[i * partition_size: (i + 1) * partition_size]
                partitions.append(partition)

            scores = []
            for i in range(n_partitions):
                partition_score = self.compute_partition_generalisation_score(partitions, i, n_percentiles, method)
                scores.append(partition_score)
            score = sum(scores) / len(scores)

        else:
            anomaly_scores = tf.sort(anomaly_scores, direction="DESCENDING")
            top_index = tf.cast(tf.cast(tf.shape(anomaly_scores)[0] - 1, tf.float32) * 0.95, tf.int32)
            anomaly_scores = anomaly_scores[top_index:]
            score = - tf.math.log(tf.math.reduce_std(anomaly_scores))

        return score

    @staticmethod
    def compute_partition_generalisation_score(partitions: List[tf.Tensor],
                                               partition_index: int,
                                               n_percentiles=5,
                                               method="kld") -> float:
        reference_partition = partitions[partition_index]
        main_partition = [partitions[i] for i in range(len(partitions)) if i != partition_index]
        main_partition = tf.concat(main_partition, axis=0)

        if method in ["kld", "pf", "corr", "corr_bis"]:
            main_partition = tf.sort(main_partition, direction="ASCENDING")

        if method in ["kld", "lt5", "corr", "corr_bis"]:
            top_index_percent = (100 - n_percentiles) / 100
            main_size = tf.shape(main_partition)[0]
            main_top_index = tf.cast(tf.cast(main_size - 1, tf.float32) * top_index_percent, tf.int32)
            main_partition = main_partition[main_top_index:]

            reference_partition = tf.sort(reference_partition, direction="ASCENDING")
            ref_top_index = tf.cast(tf.cast(tf.shape(reference_partition)[0] - 1, tf.float32) * top_index_percent,
                                    tf.int32)
            reference_partition = reference_partition[ref_top_index:]

        if method == "kld":
            main_loc = tf.reduce_mean(main_partition)
            main_scale = tf.math.reduce_std(main_partition)
            # noinspection PyUnresolvedReferences
            main_distribution = tfp.distributions.Normal(loc=main_loc, scale=main_scale)

            ref_loc = tf.reduce_mean(reference_partition)
            ref_scale = tf.math.reduce_std(reference_partition)
            # noinspection PyUnresolvedReferences
            ref_distribution = tfp.distributions.Normal(loc=ref_loc, scale=ref_scale)

            score = -tf.math.log(main_distribution.kl_divergence(ref_distribution))

        elif method in ["llh", "lt5"]:
            main_loc = tf.reduce_mean(main_partition)
            main_scale = tf.math.reduce_std(main_partition)
            # noinspection PyUnresolvedReferences
            main_distribution = tfp.distributions.Normal(loc=main_loc, scale=main_scale)

            score = tf.reduce_mean(main_distribution.prob(reference_partition))

        elif method == "pf":
            max_index = tf.cast(tf.shape(main_partition)[0] - 1, tf.float32)

            percentiles_scores = []
            for i in range(100 - n_percentiles, 100):
                i = i / 100
                threshold_index = tf.cast(max_index * i, tf.int32)
                threshold = main_partition[threshold_index]
                above_threshold = reference_partition > threshold
                above_threshold_percent = tf.reduce_mean(tf.cast(above_threshold, tf.float32))
                above_threshold_percent = above_threshold_percent.numpy()
                percentiles_score = abs(above_threshold_percent + i - 1.0)
                percentiles_scores.append(percentiles_score)

            score = - np.log(sum(percentiles_scores))

        elif method == "corr":
            reference_partition = reference_partition.numpy()
            main_partition = main_partition.numpy()[-reference_partition.size:]

            score, _ = pearsonr(main_partition, reference_partition)

        elif method == "corr_bis":
            reference_partition = reference_partition.numpy()
            main_partition = main_partition.numpy()

            x = np.arange(reference_partition.size) / (reference_partition.size - 1)
            y = reference_partition
            interpolation_function = interp1d(x, y)
            new_x = np.arange(main_partition.size) / (main_partition.size - 1)
            reference_partition = interpolation_function(new_x)

            score, _ = pearsonr(main_partition, reference_partition)

        else:
            raise ValueError(method)

        if isinstance(score, tf.Tensor):
            score = score.numpy()

        return score


class KitsuneTestSet(object):
    def __init__(self, datasets: List[str], log_dir: str, score_mode: str, use_log_score: bool):
        self.datasets = datasets
        self.log_dir = log_dir
        self.score_mode = score_mode
        self.use_log_score = use_log_score

        config_filepath = "./protocols/configs/packet/Kitsune.json"
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

        self.model_name = self.config["model_architecture"]
        self.epochs = self.config["epochs"]
        self.code_size = self.config["code_size"]

        self.backup_folder = os.path.join(self.log_dir, "packet_weights_backup", self.model_name)
        self.test_results_folder = os.path.join(self.log_dir, "kitsune_test_results", self.model_name)

    def copy_weights(self, backup: bool):
        print("======" * 10)
        print("--- Copying latest weights for testing ---")

        weights_copies = {}

        for dataset in self.datasets:
            dataset_folder = os.path.join(self.log_dir, "packet", dataset)
            train_folder = os.path.join(dataset_folder, "train", self.model_name)
            weights_folder = get_last_run_sub_folder(train_folder)
            weights_id = "weights_{:03d}".format(self.epochs)
            weights_filenames = [filename for filename in os.listdir(weights_folder) if filename.startswith(weights_id)]
            if len(weights_filenames) == 0:
                raise ValueError("Could not find weight at epoch {} for model {} in {}".format(self.epochs,
                                                                                               self.model_name,
                                                                                               weights_folder))
            else:
                print("> Copying {} `{}` from `{}` to `{}`.".format(len(weights_filenames), weights_id, weights_folder,
                                                                    dataset_folder))

            for filename in weights_filenames:
                origin = os.path.join(weights_folder, filename)
                target = os.path.join(dataset_folder, filename)
                copyfile(origin, target)
                weights_copies[target] = dataset

        if backup:
            timestamp = str(int(time()))
            current_backup_folder = os.path.join(self.backup_folder, str(self.code_size), timestamp)
            os.makedirs(current_backup_folder)
            print("> Backing up weights and config to {}".format(current_backup_folder))

            config_backup_filepath = os.path.join(current_backup_folder, "Kitsune.json")
            with open(config_backup_filepath, 'w') as config_file:
                json.dump(self.config, config_file)

            for origin, dataset in weights_copies.items():
                dataset_folder = os.path.join(current_backup_folder, dataset)
                if not os.path.exists(dataset_folder):
                    os.makedirs(dataset_folder)
                target = os.path.join(dataset_folder, os.path.basename(origin))
                copyfile(origin, target)

        print("======" * 10)

    def run_train_protocols(self, resume_from: int):
        for dataset in self.datasets:
            run_protocol(dataset, mode="train", epoch=resume_from, log_dir=self.log_dir)
            keras_backend.clear_session()

    def run_test_protocols(self):
        timestamp = str(int(time()))
        test_results_folder = os.path.join(self.test_results_folder, str(self.code_size), timestamp)
        test_results_filepath = os.path.join(test_results_folder, "anomaly_detection_results.csv")

        header_written = False
        for dataset in self.datasets:
            test = KitsuneTest(dataset_name=dataset,
                               log_dir=self.log_dir,
                               initial_epoch=self.epochs,
                               score_mode=self.score_mode)
            detection_scores = test.run_test(self.use_log_score)

            if not header_written:
                os.makedirs(test_results_folder)
                with open(test_results_filepath, "w") as results_file:
                    metrics = list(detection_scores.keys())
                    print("Dataset", *metrics, sep=",", file=results_file)
                header_written = True

            with open(test_results_filepath, "a") as results_file:
                values = list(detection_scores.values())
                print(dataset, *values, sep=",", file=results_file, flush=True)

            keras_backend.clear_session()

    def evaluate_code_sizes_generalization(self, evaluated_configs: List[str] = None):
        evaluated_configs = [int(evaluated_config) for evaluated_config in evaluated_configs]
        results_filepath = os.path.join(self.log_dir, "packet_configs_scores_{}.csv".format(int(time())))
        methods_used = ["corr"]
        scores_used = ["Score ({})".format(method) for method in methods_used]
        with open(results_filepath, "w") as results_file:
            print("Dataset", "Code size", *scores_used, sep=",", file=results_file)

        for dataset in self.datasets:
            candidates = sorted([int(candidate) for candidate in os.listdir(self.backup_folder)])
            if evaluated_configs is not None:
                candidates = [candidate for candidate in candidates if candidate in evaluated_configs]

            for candidate in candidates:
                candidate_folder = self.get_config_backup_folder(candidate)
                last_run_folder = get_last_run_sub_folder(candidate_folder)
                config_filepath = os.path.join(last_run_folder, "Kitsune.json")
                with open(config_filepath, "r") as config_file:
                    config = json.load(config_file)

                test = KitsuneTest(dataset_name=dataset,
                                   log_dir=self.log_dir,
                                   initial_epoch=self.epochs,
                                   config=config,
                                   load_weights=False,
                                   score_mode=self.score_mode,
                                   )

                weights_path = os.path.join(last_run_folder, dataset, "weights_{:03d}".format(self.epochs))
                checkpoint = tf.train.Checkpoint(test.model)
                checkpoint.restore(weights_path).expect_partial()

                packets = tf.convert_to_tensor(test.get_packets()[:test.train_samples_count], dtype=tf.float32)
                anomaly_scores = test.compute_anomaly_scores(inputs=packets, samples_length=test.sample_length,
                                                             score_function=test.compute_base_anomaly_score,
                                                             use_log=self.use_log_score)

                method_scores = []
                for method in methods_used:
                    method_score = test.compute_generalisation_score(method=method, anomaly_scores=anomaly_scores)
                    method_scores.append(method_score)

                method_scores_txt = ["Score {} : {}".format(method, method_score)
                                     for method, method_score in zip(methods_used, method_scores)]
                print("Dataset : {}".format(dataset), "Code size : {}".format(candidate), *method_scores_txt, sep=" | ")
                with open(results_filepath, "a") as results_file:
                    print(dataset, candidate, *method_scores, sep=",", file=results_file)

                keras_backend.clear_session()

        last_result_filepath = os.path.join(self.log_dir, "last_packet_configs_scores.csv")
        copyfile(results_filepath, last_result_filepath)

    def get_config_backup_folder(self, config_id) -> str:
        return os.path.join(self.backup_folder, str(config_id))

    def get_last_backup_run_config_filepath(self, config_id) -> str:
        candidate_folder = self.get_config_backup_folder(config_id)
        last_run_folder = get_last_run_sub_folder(candidate_folder)
        config_filepath = os.path.join(last_run_folder, "Kitsune.json")
        return config_filepath


def get_last_run_sub_folder(folder: str) -> str:
    last_run = sorted(os.listdir(folder))[-1]
    last_run_folder = os.path.join(folder, last_run)
    return last_run_folder


def main():
    default_datasets = [
        "Active Wiretap",
        "ARP MitM",
        "Fuzzing",
        "Mirai Botnet",
        "OS Scan",
        "SSDP Flood",
        "SSL Renegotiation",
        "SYN DoS",
        "Video Injection"
    ]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_dir", default="../logs/AEA")
    arg_parser.add_argument("--train", action="store_true")
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--evaluate_configs", nargs="*")
    arg_parser.add_argument("--no_backup", action="store_true")
    arg_parser.add_argument("--no_weight_copy", action="store_true")
    arg_parser.add_argument("--datasets", nargs="+", required=False, default=default_datasets)
    arg_parser.add_argument("--resume_from", default=0)
    arg_parser.add_argument("--copy_from", default=None)
    arg_parser.add_argument("--score_mode", default="auto")
    arg_parser.add_argument("--use_log_score", action="store_true")

    args = arg_parser.parse_args()
    log_dir: str = args.log_dir
    backup: bool = not args.no_backup
    weight_copy: bool = not args.no_weight_copy
    evaluated_configs: Optional[List[str]] = args.evaluate_configs
    evaluate_configs = evaluated_configs is not None
    datasets: List[str] = args.datasets
    resume_from: int = int(args.resume_from)
    score_mode: str = args.score_mode
    use_log_score: bool = args.use_log_score

    test_set = KitsuneTestSet(datasets, log_dir, score_mode, use_log_score)

    if args.train:
        test_set.run_train_protocols(resume_from)

    if args.test:
        if weight_copy:
            test_set.copy_weights(backup)
        test_set.run_test_protocols()

    if evaluate_configs:
        test_set.evaluate_code_sizes_generalization(evaluated_configs)


if __name__ == "__main__":
    main()
