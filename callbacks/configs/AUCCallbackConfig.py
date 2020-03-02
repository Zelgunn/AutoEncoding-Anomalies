from tensorflow.python.keras.callbacks import TensorBoard
from typing import List, Callable, Union

from callbacks import AUCCallback
from anomaly_detection import IOCompareModel
from datasets import DatasetLoader
from modalities import Pattern


class AUCCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 labels_length: int,
                 prefix: str,
                 metrics: Union[List[Union[str, Callable]], Union[str, Callable]] = "mse",
                 epoch_freq: int = 1,
                 sample_count: int = 512,
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.labels_length = labels_length
        self.prefix = prefix
        self.metrics = metrics
        self.epoch_freq = epoch_freq
        self.sample_count = sample_count

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    seed=None,
                    ) -> AUCCallback:
        raw_predictions_model = IOCompareModel(self.autoencoder,
                                               metrics=self.metrics,
                                               name="{}AutoencoderRawPredictionsModel".format(self.prefix))

        return AUCCallback.from_subset(predictions_model=raw_predictions_model,
                                       tensorboard=tensorboard,
                                       test_subset=dataset_loader.test_subset,
                                       pattern=self.pattern,
                                       labels_length=self.labels_length,
                                       samples_count=self.sample_count,
                                       epoch_freq=self.epoch_freq,
                                       batch_size=4,
                                       prefix=self.prefix,
                                       seed=seed)
