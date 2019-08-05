import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, List, Callable

from anomaly_detection import RawPredictionsLayer
from datasets import DatasetLoader, SubsetLoader
from modalities import Pattern


class AnomalyDetector(Model):
    def __init__(self,
                 inputs: Union[List[tf.Tensor], tf.Tensor],
                 output: tf.Tensor,
                 ground_truth: tf.Tensor,
                 metric: Union[str, Callable] = "mse",
                 **kwargs
                 ):

        super(AnomalyDetector, self).__init__(**kwargs)

        predictions = RawPredictionsLayer(metric=metric)([output, ground_truth])
        outputs = [predictions]

        # region Labels
        if not isinstance(inputs, list):
            inputs = [inputs]

        labels_input_layer = Input(shape=[None, 2], dtype=tf.float32, name="labels_input_layer")
        labels_output_layer = Lambda(tf.identity, name="labels_identity")(labels_input_layer)
        inputs.append(labels_input_layer)
        outputs.append(labels_output_layer)
        # endregion

        self._init_graph_network(inputs=inputs, outputs=outputs)

    def predict_anomalies_on_sample(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    sample_index: int,
                                    stride: int,
                                    normalize_predictions=False,
                                    max_steps_count=100000
                                    ):
        dataset = subset.make_source_browser(pattern, sample_index, stride)
        predictions, labels = self.predict(dataset, steps=max_steps_count)
        labels = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
        labels = np.any(labels, axis=-1)

        if normalize_predictions:
            predictions = (predictions - predictions.min()) / predictions.max()

        return predictions, labels

    def predict_anomalies_on_subset(self,
                                    subset: SubsetLoader,
                                    pattern: Pattern,
                                    stride: int,
                                    pre_normalize_predictions: bool,
                                    max_samples=10):
        predictions, labels = [], []

        sample_count = min(max_samples, len(subset.subset_folders)) if max_samples > 0 else len(subset.subset_folders)
        print("Making predictions for {} videos".format(sample_count))

        for sample_index in range(sample_count):
            sample_name = subset.subset_folders[sample_index]
            print("Predicting on sample n{}/{} ({})".format(sample_index + 1, sample_count, sample_name))
            sample_results = self.predict_anomalies_on_sample(subset, pattern,
                                                              sample_index, stride,
                                                              normalize_predictions=pre_normalize_predictions)
            sample_predictions, sample_labels = sample_results
            predictions.append(sample_predictions)
            labels.append(sample_labels)

        return predictions, labels

    def predict_anomalies(self,
                          dataset: DatasetLoader,
                          pattern: Pattern,
                          stride=1,
                          pre_normalize_predictions=True,
                          max_samples=10):
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

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        return predictions, labels, lengths

    @staticmethod
    def evaluate_predictions(predictions: np.ndarray,
                             labels: np.ndarray,
                             lengths: np.ndarray = None,
                             output_figure_filepath: str = None,
                             tensorboard: TensorBoard = None,
                             epochs_seen=0):
        if output_figure_filepath is not None:
            plt.plot(np.mean(predictions, axis=1), linewidth=0.2)
            plt.savefig(output_figure_filepath, dpi=1000)
            plt.gca().fill_between(np.arange(len(labels)), 0, labels, alpha=0.5)
            # plt.plot(labels, alpha=0.75, linewidth=0.2)
            if lengths is not None:
                lengths_splits = np.zeros(shape=predictions.shape, dtype=np.float32)
                lengths_splits[lengths - 1] = 1.0
                plt.plot(lengths_splits, alpha=0.5, linewidth=0.05)
            plt.savefig(output_figure_filepath[:-4] + "_labeled.png", dpi=1000)
            plt.clf()  # clf = clear figure

        roc = tf.metrics.AUC(curve="ROC")
        pr = tf.metrics.AUC(curve="PR")

        if predictions.ndim == 2 and labels.ndim == 1:
            predictions = predictions.mean(axis=-1)

        roc.update_state(labels, predictions)
        pr.update_state(labels, predictions)

        roc_result = roc.result()
        pr_result = pr.result()

        if tensorboard is not None:
            # noinspection PyProtectedMember
            with tensorboard._get_writer(tensorboard._train_run_name).as_default():
                tf.summary.scalar(name="ROC_AUC", data=roc_result, step=epochs_seen)
                tf.summary.scalar(name="PR_AUC", data=pr_result, step=epochs_seen)

        return roc_result, pr_result

    def predict_and_evaluate(self,
                             dataset: DatasetLoader,
                             pattern: Pattern,
                             log_dir: str,
                             stride=1,
                             tensorboard: TensorBoard = None,
                             epochs_seen=0,
                             pre_normalize_predictions=True,
                             max_samples=-1,
                             ):
        predictions, labels, lengths = self.predict_anomalies(dataset=dataset,
                                                              pattern=pattern,
                                                              stride=stride,
                                                              pre_normalize_predictions=pre_normalize_predictions,
                                                              max_samples=max_samples)
        graph_filepath = os.path.join(log_dir, "Anomaly_score.png")
        roc, pr = self.evaluate_predictions(predictions=predictions,
                                            labels=labels,
                                            lengths=lengths,
                                            output_figure_filepath=graph_filepath,
                                            tensorboard=tensorboard,
                                            epochs_seen=epochs_seen)
        print("Anomaly_score : ROC = {} | PR = {}".format(roc, pr))
        with open(os.path.join(log_dir, "anomaly_detection_scores.txt"), 'w') as file:
            file.write("ROC = {} | PR = {}".format(roc, pr))
        return roc, pr

    def compute_output_signature(self, input_signature):
        raise NotImplementedError
