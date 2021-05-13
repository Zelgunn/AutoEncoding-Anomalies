from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard
from typing import List, Callable, Union

from callbacks import AUCCallback
from anomaly_detection import IOCompareModel
from datasets import DatasetLoader
from modalities import Pattern
from custom_tf_models.utils import LambdaModel
from misc_utils.general import get_model_inputs_count


class AUCCallbackConfig(object):
    def __init__(self,
                 base_model: Union[Model, Callable],
                 pattern: Pattern,
                 labels_length: int,
                 prefix: str,
                 epoch_freq: int = 1,
                 batch_size: int = 4,
                 sample_count: int = 128,
                 convert_to_io_compare_model=False,
                 io_compare_metrics: Union[List[Union[str, Callable]], Union[str, Callable]] = "mse"
                 ):

        if convert_to_io_compare_model:
            predictions_model = IOCompareModel(base_model,
                                               metrics=io_compare_metrics,
                                               name="{}AutoencoderRawPredictionsModel".format(prefix))
        elif not isinstance(base_model, Model):
            predictions_model = LambdaModel(function=base_model, name="{}RawPredictionsModel".format(prefix))
        else:
            predictions_model = base_model

        self.base_model = base_model
        self.predictions_model = predictions_model
        self.pattern = pattern
        self.labels_length = labels_length
        self.prefix = prefix
        self.epoch_freq = epoch_freq
        self.batch_size = batch_size
        self.sample_count = sample_count

    @property
    def modality_count(self) -> int:
        return get_model_inputs_count(self.base_model)

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    seed=None,
                    ) -> AUCCallback:
        return AUCCallback.from_subset(predictions_model=self.predictions_model,
                                       tensorboard=tensorboard,
                                       test_subset=dataset_loader.test_subset,
                                       pattern=self.pattern,
                                       labels_length=self.labels_length,
                                       modality_count=self.modality_count,
                                       samples_count=self.sample_count,
                                       epoch_freq=self.epoch_freq,
                                       batch_size=self.batch_size,
                                       prefix=self.prefix,
                                       seed=seed)
