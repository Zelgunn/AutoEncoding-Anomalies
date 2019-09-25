import tensorflow as tf
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, List, Callable, Dict, Any, Tuple

from anomaly_detection import RawPredictionsModel
from datasets import DatasetLoader, SubsetLoader
from modalities import Pattern


class AnomalyDetector(Model):
    def __init__(self,
                 autoencoder: Callable,
                 output_length: int,
                 metrics: List[Union[str, Callable]] = "mse",
                 **kwargs
                 ):

        super(AnomalyDetector, self).__init__(**kwargs)

        self.raw_prediction_model = RawPredictionsModel(autoencoder=autoencoder,
                                                        output_length=output_length,
                                                        metrics=metrics)

        if not (isinstance(metrics, list) or isinstance(metrics, tuple)):
            metrics = [metrics]

        self.anomaly_metrics_names = []
        for metric in metrics:
            metric_name = metric if isinstance(metric, str) else metric.__name__
            self.anomaly_metrics_names.append(metric_name)

    def call(self, inputs, training=None, mask=None):
        raw_predictions = self.raw_prediction_model(inputs)
        return raw_predictions

    def predict_anomalies_on_sample(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    sample_index: int,
                                    stride: int,
                                    normalize_predictions=False,
                                    ):
        dataset = subset.make_source_browser(pattern, sample_index, stride)

        get_outputs_from_inputs = len(dataset.element_spec) == 2

        predictions, labels = None, []
        for sample in dataset:
            if get_outputs_from_inputs:
                sample_inputs, sample_labels = sample
                sample_outputs = sample_inputs
            else:
                sample_inputs, sample_outputs, sample_labels = sample

            sample_predictions = self([sample_inputs, sample_outputs])

            labels.append(sample_labels)
            if predictions is None:
                predictions = [[metric_prediction] for metric_prediction in sample_predictions]
            else:
                for i in range(len(predictions)):
                    predictions[i].append(sample_predictions[i])

        predictions = [np.concatenate(metric_prediction, axis=0) for metric_prediction in predictions]
        labels = np.concatenate(labels, axis=0)

        # *predictions, labels = self.predict(dataset, steps=max_steps_count)
        labels = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
        labels = np.any(labels, axis=-1)

        if normalize_predictions:
            for i in range(len(predictions)):
                metric_pred = predictions[i]
                metric_pred = (metric_pred - metric_pred.min()) / (metric_pred.max() - metric_pred.min())
                predictions[i] = metric_pred

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    stride: int,
                                    pre_normalize_predictions: bool,
                                    max_samples=10
                                    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        labels = []
        predictions = [[] for _ in range(self.metrics_count)]

        sample_count = min(max_samples, len(subset.subset_folders)) if max_samples > 0 else len(subset.subset_folders)
        print("Making predictions for {} videos".format(sample_count))

        for sample_index in range(sample_count):
            sample_name = subset.subset_folders[sample_index]
            print("Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name))
            sample_results = self.predict_anomalies_on_sample(subset, pattern,
                                                              sample_index, stride,
                                                              normalize_predictions=pre_normalize_predictions)
            sample_predictions, sample_labels = sample_results
            for i in range(self.metrics_count):
                predictions[i].append(sample_predictions[i])
            labels.append(sample_labels)

        return predictions, labels

    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          pattern: Pattern,
                          stride=1,
                          pre_normalize_predictions=True,
                          max_samples=10
                          ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        predictions, labels = self.predict_anomalies_on_subset(subset=dataset.test_subset,
                                                               pattern=pattern,
                                                               stride=stride,
                                                               pre_normalize_predictions=pre_normalize_predictions,
                                                               max_samples=max_samples)

        lengths = np.empty(shape=[len(labels)], dtype=np.int32)
        for i in range(len(labels)):
            lengths[i] = len(labels[i])
            if i > 0:
                lengths[i] += lengths[i - 1]

        final_predictions = []
        for i in range(self.metrics_count):
            metric_pred = np.concatenate(predictions[i])
            metric_pred_min = metric_pred.min()
            metric_pred = (metric_pred - metric_pred_min) / (metric_pred.max() - metric_pred_min)
            final_predictions.append(metric_pred)

        labels = np.concatenate(labels)

        return final_predictions, labels, lengths

    @staticmethod
    def evaluate_predictions(predictions: List[np.ndarray],
                             labels: np.ndarray):

        roc_results, pr_results = [], []
        for i in range(len(predictions)):
            metric_roc_result, metric_pr_result = AnomalyDetector.evaluate_metric_predictions(predictions[i], labels)
            roc_results.append(metric_roc_result)
            pr_results.append(metric_pr_result)

        return roc_results, pr_results

    @staticmethod
    def evaluate_metric_predictions(predictions: np.ndarray,
                                    labels: np.ndarray
                                    ):
        roc = tf.metrics.AUC(curve="ROC")
        pr = tf.metrics.AUC(curve="PR")

        if predictions.ndim == 2 and labels.ndim == 1:
            predictions = predictions.mean(axis=-1)

        roc.update_state(labels, predictions)
        pr.update_state(labels, predictions)

        roc_result = roc.result()
        pr_result = pr.result()
        return roc_result, pr_result

    def plot_predictions(self,
                         predictions: List[np.ndarray],
                         labels: np.ndarray,
                         output_figure_filepath: str,
                         lengths: np.ndarray = None):

        for i in range(len(predictions)):
            plt.plot(np.mean(predictions[i], axis=1), linewidth=0.2)
        plt.legend(self.anomaly_metrics_names)

        plt.savefig(output_figure_filepath, dpi=1000)
        plt.gca().fill_between(np.arange(len(labels)), 0, labels, alpha=0.5)

        if lengths is not None:
            lengths_splits = np.zeros(shape=predictions[0].shape, dtype=np.float32)
            lengths_splits[lengths - 1] = 1.0
            plt.plot(lengths_splits, alpha=0.5, linewidth=0.05)

        plt.savefig(output_figure_filepath[:-4] + "_labeled.png", dpi=1000)
        plt.clf()  # clf = clear figure

    def save_evaluation_results(self,
                                log_dir: str,
                                roc_results: List[float],
                                pr_results: List[float],
                                additional_config: Dict[str, any] = None):
        with open(os.path.join(log_dir, "anomaly_detection_scores.txt"), 'w') as file:
            for i in range(self.metrics_count):
                file.write("{metric}) ROC = {roc} | PR = {pr}\n".format(metric=self.anomaly_metrics_names[i],
                                                                        roc=roc_results[i],
                                                                        pr=pr_results[i]))
            if additional_config is not None:
                for key, value in additional_config.items():
                    file.write("{}: {}\n".format(key, value))

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             pattern: Pattern,
                             log_dir: str,
                             stride=1,
                             pre_normalize_predictions=True,
                             max_samples=-1,
                             additional_config: Dict[str, Any] = None,
                             ):
        predictions, labels, lengths = self.predict_anomalies(dataset=dataset,
                                                              pattern=pattern,
                                                              stride=stride,
                                                              pre_normalize_predictions=pre_normalize_predictions,
                                                              max_samples=max_samples)

        roc, pr = self.evaluate_predictions(predictions=predictions, labels=labels)

        figures_filepath = os.path.join(log_dir, "anomaly_score.png")
        self.plot_predictions(predictions=predictions,
                              labels=labels,
                              output_figure_filepath=figures_filepath,
                              lengths=lengths)

        for i in range(self.metrics_count):
            print("Anomaly_score ({}): ROC = {} | PR = {}".format(self.anomaly_metrics_names[i], roc[i], pr[i]))

        additional_config["stride"] = stride
        additional_config["pre-normalize predictions"] = pre_normalize_predictions
        self.save_evaluation_results(log_dir=log_dir,
                                     roc_results=roc,
                                     pr_results=pr,
                                     additional_config=additional_config)
        return roc, pr

    def compute_output_signature(self, input_signature):
        raise NotImplementedError

    @property
    def metrics_count(self) -> int:
        return len(self.anomaly_metrics_names)
