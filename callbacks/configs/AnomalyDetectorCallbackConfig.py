import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from typing import List, Callable, Union, Optional

from callbacks import AnomalyDetectorCallback
from datasets import DatasetLoader
from modalities import Pattern
from custom_tf_models import AE


class AnomalyDetectorCallbackConfig(object):
    def __init__(self,
                 autoencoder: Union[Callable, AE],
                 pattern: Pattern,
                 compare_metrics: Optional[List[Union[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]]] = "mse",
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 stride: int = 1,
                 epoch_freq: int = 1,
                 pre_normalize_predictions: bool = True,
                 max_samples: int = -1,
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.compare_metrics = compare_metrics
        self.additional_metrics = additional_metrics
        self.stride = stride
        self.epoch_freq = epoch_freq
        self.pre_normalize_predictions = pre_normalize_predictions
        self.max_samples = max_samples

    def to_callback(self, tensorboard: TensorBoard, dataset_loader: DatasetLoader) -> AnomalyDetectorCallback:
        return AnomalyDetectorCallback(tensorboard=tensorboard,
                                       autoencoder=self.autoencoder,
                                       dataset=dataset_loader,
                                       pattern=self.pattern,
                                       compare_metrics=self.compare_metrics,
                                       additional_metrics=self.additional_metrics,
                                       stride=self.stride,
                                       epoch_freq=self.epoch_freq,
                                       pre_normalize_predictions=self.pre_normalize_predictions,
                                       max_samples=self.max_samples)
