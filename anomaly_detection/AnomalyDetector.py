import tensorflow as tf
from tensorflow.python.keras.models import Model
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Union, List, Callable, Dict, Any, Tuple, Sequence, Optional

from anomaly_detection import IOCompareModel
from datasets import DatasetLoader, SubsetLoader
from modalities import Pattern
from custom_tf_models import AE
from misc_utils.numpy_utils import normalize
from misc_utils.general import unpack_test_sample


class AnomalyDetector(Model):
    def __init__(self,
                 model: Union[Callable, Model, AE],
                 pattern: Pattern,
                 compare_metrics: Optional[List[Union[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]]] = "mse",
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 **kwargs
                 ):
        """

        :param model:
        :param pattern:
        :param compare_metrics:
        :param additional_metrics:
        :param kwargs:
        """
        super(AnomalyDetector, self).__init__(**kwargs)

        if (compare_metrics is None) and (additional_metrics is None):
            raise ValueError("At least `compare_metrics` or `additional_metrics` must be specified.")

        self.model = model
        self.pattern = pattern
        self.compare_metrics = compare_metrics

        if compare_metrics is not None:
            self.io_compare_model = IOCompareModel(autoencoder=model,
                                                   postprocessor=self.pattern.postprocessor,
                                                   metrics=compare_metrics)
        else:
            self.io_compare_model = None

        self.additional_metrics = to_list(additional_metrics) if additional_metrics is not None else []
        compare_metrics = to_list(compare_metrics) if compare_metrics is not None else []

        self.anomaly_metrics_names = []
        all_metrics = compare_metrics + self.additional_metrics
        for metric in all_metrics:
            metric_name = metric if isinstance(metric, str) else metric.__name__
            self.anomaly_metrics_names.append(metric_name)

        self.prog_bar = None

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs, ground_truth = inputs

        if isinstance(inputs, (list, tuple)):
            inputs = [tf.expand_dims(x, axis=0) for x in inputs]
            ground_truth = [tf.expand_dims(x, axis=0) for x in ground_truth]
        elif inputs.shape[0] != 1:
            inputs = tf.expand_dims(inputs, axis=0)
            ground_truth = tf.expand_dims(ground_truth, axis=0)

        inputs = self.pattern.process_batch(inputs)
        ground_truth = self.pattern.process_batch(ground_truth)

        if self.compare_metrics is not None:
            predictions = self.io_compare_model([inputs, ground_truth])
        else:
            predictions = []

        for additional_metric in self.additional_metrics:
            prediction = additional_metric(inputs)
            prediction = tf.reduce_mean(prediction, axis=0, keepdims=True)
            predictions.append(prediction)

        return predictions

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             log_dir: str,
                             stride=1,
                             pre_normalize_predictions=True,
                             max_samples=-1,
                             additional_config: Dict[str, Any] = None,
                             ):
        predictions, labels = self.predict_anomalies(dataset=dataset,
                                                     stride=stride,
                                                     pre_normalize_predictions=pre_normalize_predictions,
                                                     max_samples=max_samples)

        merged_predictions, merged_labels = self.merge_samples_predictions(predictions=predictions, labels=labels)
        self.save_predictions(predictions=merged_predictions, labels=merged_labels, log_dir=log_dir)
        results = self.evaluate_predictions(predictions=merged_predictions, labels=merged_labels)
        self.print_results(results)

        self.plot_predictions(predictions=predictions,
                              labels=labels,
                              log_dir=log_dir,
                              samples_names=dataset.test_subset.sample_names)

        additional_config["stride"] = stride
        additional_config["pre-normalize predictions"] = pre_normalize_predictions
        self.save_evaluation_results(log_dir=log_dir,
                                     results=results,
                                     additional_config=additional_config)
        return results

    # region Predict anomaly scores
    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          stride=1,
                          pre_normalize_predictions=True,
                          max_samples=-1
                          ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        predictions, labels = self.predict_anomalies_on_subset(subset=dataset.test_subset,
                                                               stride=stride,
                                                               pre_normalize_predictions=pre_normalize_predictions,
                                                               max_samples=max_samples)

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    stride: int,
                                    pre_normalize_predictions: bool,
                                    max_samples=-1
                                    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        labels = []
        predictions = [[] for _ in range(self.metric_count)]

        sample_count = min(max_samples, subset.sample_count) if max_samples > 0 else subset.sample_count
        print("Making predictions for {} samples".format(sample_count))

        with tqdm(total=subset.size) as prog_bar:
            self.prog_bar = prog_bar
            for sample_index in range(sample_count):
                sample_name = subset.sample_names[sample_index]
                description = "Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name)
                self.prog_bar.set_description(description)
                sample_results = self.predict_anomalies_on_sample(subset, sample_index, stride,
                                                                  normalize_predictions=pre_normalize_predictions)
                sample_predictions, sample_labels = sample_results
                for i in range(self.metric_count):
                    predictions[i].append(sample_predictions[i])
                labels.append(sample_labels)

        return predictions, labels

    def predict_anomalies_on_sample(self,
                                    subset: SubsetLoader,
                                    sample_index: int,
                                    stride: int,
                                    normalize_predictions=False,
                                    ):
        dataset = subset.make_source_browser(self.pattern, sample_index, stride=stride)

        predictions, labels = None, []

        for sample in dataset:
            sample_inputs, sample_outputs, sample_labels = unpack_test_sample(sample, self.modality_count)
            sample_predictions = self([sample_inputs, sample_outputs])

            labels.append(sample_labels)
            if predictions is None:
                predictions = [[metric_prediction] for metric_prediction in sample_predictions]
            else:
                for i in range(len(predictions)):
                    predictions[i].append(sample_predictions[i])
            if self.prog_bar is not None:
                self.prog_bar.update(stride)

        predictions = [np.concatenate(metric_prediction, axis=0) for metric_prediction in predictions]
        if (len(labels[0].shape) == 2) and (labels[0].shape[1] == 2):
            labels = self.timestamps_labels_to_frame_labels(labels)
        elif len(labels[0].shape) == 1:
            labels = np.stack(labels, axis=0)
            labels = np.any(labels > 0.5, axis=1)
        else:
            raise ValueError("Labels should either be timestamps with shape [n, 2] or [length], got {}".
                             format(labels[0].shape))

        if normalize_predictions:
            predictions = [normalize(metric_prediction) for metric_prediction in predictions]

        return predictions, labels

    # endregion

    def save_predictions(self, predictions: List[np.ndarray], labels: np.ndarray, log_dir: str):
        for i in range(self.metric_count):
            np.save(os.path.join(log_dir, "predictions_{}.npy".format(i)), predictions[i])

            if predictions[i].ndim > 1:
                predictions[i] = np.mean(predictions[i], axis=tuple(range(1, predictions[i].ndim)))

        np.save(os.path.join(log_dir, "predictions_mean.npy"), predictions)
        np.save(os.path.join(log_dir, "labels.npy"), labels)

    @staticmethod
    def merge_samples_predictions(predictions: List[List[np.ndarray]],
                                  labels: List[np.ndarray],
                                  ) -> Tuple[List[np.ndarray], np.ndarray]:
        merged_predictions = []
        metric_count = len(predictions)
        for i in range(metric_count):
            metric_pred = np.concatenate(predictions[i])
            merged_predictions.append(metric_pred)

        labels = np.concatenate(labels)
        return merged_predictions, labels

    # region Evaluate predictions
    @staticmethod
    def evaluate_predictions(predictions: List[np.ndarray],
                             labels: np.ndarray,
                             evaluation_metrics: List[str] = None,
                             ) -> Dict[str, List[float]]:
        predictions = [normalize(metric_predictions) for metric_predictions in predictions]

        results = None
        for i in range(len(predictions)):
            metric_results = AnomalyDetector.evaluate_metric_predictions(predictions[i], labels, evaluation_metrics)

            if results is None:
                results = {result_name: [] for result_name in metric_results}

            for result_name in metric_results:
                results[result_name].append(metric_results[result_name])

        return results

    @staticmethod
    def evaluate_metric_predictions(predictions: np.ndarray,
                                    labels: np.ndarray,
                                    evaluation_metrics: List[str] = None,
                                    ):
        # region Evaluation metrics used (ROC, EER, PR, Precision)
        available_metrics = ["roc", "eer", "pr", "precision"]
        if evaluation_metrics is None:
            evaluation_metrics = available_metrics
        evaluation_metrics = [evaluation_metric.lower() for evaluation_metric in evaluation_metrics]
        for evaluation_metric in evaluation_metrics:
            if evaluation_metric not in available_metrics:
                raise ValueError("Unknown evaluation metric : {}. Available evaluation metrics are : {}".format(
                    evaluation_metric, available_metrics))
        # endregion

        results = {}

        # region Normalize predictions
        if predictions.ndim > 1 and labels.ndim == 1:
            predictions = predictions.mean(axis=tuple(range(1, predictions.ndim)))
        predictions = normalize(predictions)
        # endregion

        # region ROC / EER
        if ("roc" in evaluation_metrics) or ("eer" in evaluation_metrics):
            roc = tf.metrics.AUC(curve="ROC", num_thresholds=1000)
            roc.update_state(labels, predictions)

            # region ROC
            if "roc" in evaluation_metrics:
                results["ROC"] = roc.result()
            # endregion

            # region EER
            if "eer" in evaluation_metrics:
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
        if "pr" in evaluation_metrics:
            pr = tf.metrics.AUC(curve="PR", num_thresholds=1000)
            pr.update_state(labels, predictions)
            results["PR"] = pr.result()
        # endregion

        # region Precision
        if "precision" in evaluation_metrics:
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

    # endregion

    # region Plotting
    def plot_predictions(self,
                         predictions: List[List[np.ndarray]],
                         labels: List[np.ndarray],
                         log_dir: str,
                         samples_names: List[str],
                         ):
        metrics_mins = [np.min([np.min(x) for x in metric_pred]) for metric_pred in predictions]
        metrics_maxs = [np.max([np.max(x) for x in metric_pred]) for metric_pred in predictions]

        sample_count = len(labels)
        for i in range(sample_count):
            sample_predictions = [((predictions[j][i] - metrics_mins[j]) / (metrics_maxs[j] - metrics_mins[j]))
                                  for j in range(self.metric_count)]
            self.plot_sample_predictions(predictions=sample_predictions,
                                         labels=labels[i],
                                         log_dir=log_dir,
                                         sample_name=samples_names[i])

    def plot_sample_predictions(self,
                                predictions: List[np.ndarray],
                                labels: np.ndarray,
                                log_dir: str,
                                sample_name: str,
                                linewidth=0.1,
                                include_legend=True,
                                font_size=4,
                                clear_figure=True,
                                ratio=None,
                                ):
        plt.ylim(0.0, 1.0)

        sample_length = len(labels)
        for i in range(self.metric_count):
            metric_predictions = predictions[i]
            if metric_predictions.ndim > 1:
                metric_predictions = np.mean(metric_predictions, axis=tuple(range(1, metric_predictions.ndim)))
            plt.plot(1.0 - metric_predictions, linewidth=linewidth)

        if include_legend:
            plt.legend(self.anomaly_metrics_names,
                       loc="center", bbox_to_anchor=(0.5, -0.4), fontsize=font_size,
                       fancybox=True, shadow=True)

        if ratio is None:
            ratio = np.log(sample_length + 1) * 0.75

        adjust_figure_aspect(plt.gcf(), ratio)
        dpi = (np.sqrt(sample_length) + 100) * 4

        self.plot_labels(labels)

        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        labeled_filepath = os.path.join(log_dir, "{}.png".format(sample_name))
        plt.savefig(labeled_filepath, dpi=dpi, bbox_inches='tight')

        if clear_figure:
            plt.clf()

    @staticmethod
    def plot_labels(labels: np.ndarray):
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

    # endregion

    def print_results(self, results: Dict[str, List], print_fn=None):
        if print_fn is None:
            print_fn = print

        for i in range(self.metric_count):
            metric_results_string = "Anomaly_score ({}):".format(self.anomaly_metrics_names[i])
            for result_name, result_values in results.items():
                metric_results_string += " {} = {:.3f} |".format(result_name, result_values[i])
            print_fn(metric_results_string)

    def save_evaluation_results(self,
                                log_dir: str,
                                results: Dict[str, List[Union[tf.Tensor, float]]],
                                additional_config: Dict[str, any] = None):
        with open(os.path.join(log_dir, "anomaly_detection_scores.txt"), 'w') as file:
            for i in range(self.metric_count):
                line = "{})".format(self.anomaly_metrics_names[i])
                for result_name, result_values in results.items():
                    value = result_values[i]
                    # value = round(float(value), 3)
                    line += " {} = {} |".format(result_name, value)
                line += "\n"
                file.write(line)

            file.write("\n")

            if additional_config is not None:
                for key, value in additional_config.items():
                    file.write("{}: {}\n".format(key, value))

    # region Latent space visualization

    def compute_latent_codes_on_dataset(self,
                                        dataset: DatasetLoader,
                                        stride=1,
                                        ) -> Tuple[tf.Tensor, Dict[str, Sequence]]:
        if not hasattr(self.model, "encode"):
            raise ValueError("self.autoencoder must have an `encode` method.")

        train_latent_codes, train_samples_infos = self.compute_latent_codes_on_subset(subset=dataset.train_subset,
                                                                                      stride=stride)

        test_latent_codes, test_samples_infos = self.compute_latent_codes_on_subset(subset=dataset.test_subset,
                                                                                    stride=stride)

        latent_codes = tf.concat([train_latent_codes, test_latent_codes], axis=0)
        samples_infos = {}
        for info_names in train_samples_infos:
            infos = np.concatenate([train_samples_infos[info_names], test_samples_infos[info_names]], axis=0)
            samples_infos[info_names] = infos

        return latent_codes, samples_infos

    def compute_latent_codes_on_subset(self,
                                       subset: SubsetLoader,
                                       stride=1,
                                       ) -> Tuple[tf.Tensor, Dict[str, Sequence]]:
        latent_codes: Union[tf.Tensor, List[tf.Tensor]] = []
        names, labels = [], []

        sample_count = subset.sample_count
        print("Computing latent codes for {} samples".format(sample_count))

        for sample_index in range(sample_count):
            sample_name = subset.sample_names[sample_index]
            print("Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name))

            sample_latent_codes, sample_labels = self.compute_latent_codes_on_sample(subset, sample_index, stride)
            latent_codes.append(sample_latent_codes)

            names += [sample_name] * len(sample_latent_codes)
            labels.append(sample_labels)

        latent_codes: tf.Tensor = tf.concat(latent_codes, axis=0)
        labels = np.concatenate(labels, axis=0)
        samples_infos = {"__name__": names, "__label__": labels}
        return latent_codes, samples_infos

    def compute_latent_codes_on_sample(self,
                                       subset: SubsetLoader,
                                       sample_index: int,
                                       stride=1,
                                       ) -> Tuple[tf.Tensor, np.ndarray]:
        dataset = subset.make_source_browser(self.pattern, sample_index, stride=stride)

        latent_codes, labels = [], []

        for sample in dataset:
            sample_inputs, _, sample_labels = unpack_test_sample(sample, self.modality_count)

            sample_inputs = tf.expand_dims(sample_inputs, axis=0)
            sample_inputs = self.pattern.process_batch(sample_inputs)

            latent_code = self.model.encode(sample_inputs)
            latent_code = tf.reduce_mean(latent_code, axis=[1, 2, 3])
            latent_code = tf.reshape(latent_code, shape=[-1])
            latent_codes.append(latent_code)
            labels.append(sample_labels)

        latent_codes = tf.stack(latent_codes, axis=0)
        labels = self.timestamps_labels_to_frame_labels(labels)
        return latent_codes, labels

    # endregion

    @property
    def metric_count(self) -> int:
        return len(self.anomaly_metrics_names)

    @property
    def modality_count(self) -> int:
        return len(self.pattern) - 1

    @staticmethod
    def timestamps_labels_to_frame_labels(labels: Union[List[np.ndarray], List[tf.Tensor]]) -> np.ndarray:
        max_label_length = max([len(label) for label in labels])
        labels_array = np.ones(shape=(len(labels), max_label_length, 2), dtype=np.float32)
        for i, label in enumerate(labels):
            labels_array[i][:len(label)] = label
        labels = labels_array

        start, end = labels[..., 0], labels[..., 1]
        start_before_half = start <= 0.5
        end_after_half = end >= 0.5
        not_equal = start != end

        labels = np.logical_and(start_before_half, end_after_half)
        labels = np.logical_and(labels, not_equal)
        labels = np.any(labels, axis=1)

        # labels = SubsetLoader.timestamps_labels_to_frame_labels(labels, frame_count)
        # mask = np.zeros_like(labels)
        # mask[:, (frame_count - 1) // 2] = 1
        # labels = np.sum(labels.numpy() * mask, axis=-1) >= 1
        return labels

    def get_config(self):
        return {
            "autoencoder": self.model,
            "pattern": self.pattern,
            "compare_metrics": self.compare_metrics,
            "additional_metrics": self.additional_metrics,
        }


def adjust_figure_aspect(fig, aspect=1.0):
    """
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    """
    x_size, y_size = fig.get_size_inches()
    minsize = min(x_size, y_size)
    x_lim = .4 * minsize / x_size
    y_lim = .4 * minsize / y_size

    if aspect < 1:
        x_lim *= aspect
    else:
        y_lim /= aspect

    fig.subplots_adjust(left=.5 - x_lim,
                        right=.5 + x_lim,
                        bottom=.5 - y_lim,
                        top=.5 + y_lim)


def to_list(x: Union[List, Tuple, Any]) -> Union[List, Tuple]:
    if not (isinstance(x, list) or isinstance(x, tuple)):
        x = [x]
    return x
