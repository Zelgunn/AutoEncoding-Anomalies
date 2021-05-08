import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context
import numpy as np
from time import time
from typing import Union, Callable, Optional, List, Tuple, Dict

from callbacks import TensorBoardPlugin
from anomaly_detection import AnomalyDetector
from datasets import DatasetLoader
from modalities import Pattern
from custom_tf_models import AE


class AnomalyDetectorCallback(TensorBoardPlugin):
    def __init__(self,
                 tensorboard: TensorBoard,
                 autoencoder: Union[Callable, AE],
                 dataset: DatasetLoader,
                 pattern: Pattern,
                 compare_metrics: Optional[List[Union[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]]] = "mse",
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 stride: int = 1,
                 epoch_freq: int = 1,
                 pre_normalize_predictions: bool = True,
                 max_samples: int = -1,
                 name="AnomalyDetector",
                 **kwargs):
        super(AnomalyDetectorCallback, self).__init__(tensorboard=tensorboard,
                                                      update_freq="epoch",
                                                      epoch_freq=epoch_freq)

        self.anomaly_detector = AnomalyDetector(model=autoencoder,
                                                pattern=pattern,
                                                compare_metrics=compare_metrics,
                                                additional_metrics=additional_metrics,
                                                **kwargs)
        self.dataset = dataset
        self.stride = stride
        self.pre_normalize_predictions = pre_normalize_predictions
        self.max_samples = max_samples
        self.name = name

    def _write_logs(self, step: int):
        start_time = time()

        try:
            predictions, labels = self.predict_anomalies()
            predictions, labels = self.anomaly_detector.merge_samples_predictions(predictions=predictions,
                                                                                  labels=labels)
            results = self.anomaly_detector.evaluate_predictions(predictions=predictions, labels=labels,
                                                                 evaluation_metrics=["roc", "pr"])
            self.anomaly_detector.print_results(results)

            with context.eager_mode():
                with summary_ops_v2.always_record_summaries():
                    with self.validation_run_writer.as_default():
                        self.write_results(results, step=step)

        except tf.errors.InvalidArgumentError:
            print("Could not predict this time.")

        print("`{}` took {:.2f} seconds.".format(self.name, time() - start_time))

    def predict_anomalies(self) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        return self.anomaly_detector.predict_anomalies(dataset=self.dataset,
                                                       stride=self.stride,
                                                       pre_normalize_predictions=self.pre_normalize_predictions,
                                                       max_samples=self.max_samples)

    def write_results(self, results: Dict[str, List[float]], step: int):
        for i in range(self.anomaly_detector.metric_count):
            metric_name = self.anomaly_detector.anomaly_metrics_names[i]
            for evaluation_metric_name, evaluation_metric_value in results.items():
                name = "{}/{}/{}".format(self.name, metric_name, evaluation_metric_name)
                summary_ops_v2.scalar(name=name, tensor=evaluation_metric_value[i], step=step)
