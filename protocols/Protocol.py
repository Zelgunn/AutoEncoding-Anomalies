from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, TensorBoard, EarlyStopping
import time
import os
from typing import List, Callable

from anomaly_detection import AnomalyDetector, known_metrics, RawPredictionsModel
from datasets import DatasetLoader, DatasetConfig
from utils.train_utils import save_model_info
from callbacks import ImageCallback, EagerModelCheckpoint, AUCCallback
from modalities import Pattern


class ImageCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 is_train_callback: bool,
                 name: str,
                 epoch_freq: int = 1
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.name = name
        self.is_train_callback = is_train_callback
        self.epoch_freq = epoch_freq

    def to_callbacks(self,
                     tensorboard: TensorBoard,
                     dataset_loader: DatasetLoader,
                     ) -> List[ImageCallback]:
        subset = dataset_loader.train_subset if self.is_train_callback else dataset_loader.test_subset
        image_callbacks = ImageCallback.from_model_and_subset(autoencoder=self.autoencoder,
                                                              subset=subset,
                                                              pattern=self.pattern,
                                                              name=self.name,
                                                              is_train_callback=self.is_train_callback,
                                                              tensorboard=tensorboard,
                                                              epoch_freq=self.epoch_freq)
        return image_callbacks


class AUCCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 output_length: int,
                 prefix: str,
                 epoch_freq: int = 1
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.output_length = output_length
        self.prefix = prefix
        self.epoch_freq = epoch_freq

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    ) -> AUCCallback:
        raw_predictions_model = RawPredictionsModel(self.autoencoder,
                                                    output_length=self.output_length,
                                                    name="{}AutoencoderRawPredictionsModel".format(self.prefix))

        return AUCCallback.from_subset(predictions_model=raw_predictions_model,
                                       tensorboard=tensorboard,
                                       test_subset=dataset_loader.test_subset,
                                       pattern=self.pattern,
                                       samples_count=128,
                                       epoch_freq=self.epoch_freq,
                                       batch_size=4,
                                       prefix=self.prefix)


class ProtocolTrainConfig(object):
    def __init__(self,
                 batch_size: int,
                 pattern: Pattern,
                 epochs: int,
                 initial_epoch: int,
                 image_callbacks_configs: List[ImageCallbackConfig] = None,
                 auc_callbacks_configs: List[AUCCallbackConfig] = None,
                 early_stopping_metric: str = None,
                 ):
        self.batch_size = batch_size
        self.pattern = pattern
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.image_callbacks_configs = image_callbacks_configs
        self.auc_callbacks_configs = auc_callbacks_configs
        self.early_stopping_metric = early_stopping_metric


class ProtocolTestConfig(object):
    def __init__(self,
                 pattern: Pattern,
                 epoch: int,
                 output_length: int,
                 detector_stride: int,
                 pre_normalize_predictions: bool,
                 **kwargs,
                 ):
        self.pattern = pattern
        self.epoch = epoch
        self.output_length = output_length
        self.detector_stride = detector_stride
        self.pre_normalize_predictions = pre_normalize_predictions
        self.kwargs = kwargs


class Protocol(object):
    def __init__(self,
                 model: Model,
                 dataset_name: str,
                 protocol_name: str,
                 autoencoder: Callable = None,
                 model_name: str = None
                 ):
        self.model = model
        if autoencoder is None:
            autoencoder = model
        self.autoencoder = autoencoder
        if model_name is None:
            model_name = model.name
        self.model_name = model_name

        self.dataset_name = dataset_name
        self.dataset_folder = get_dataset_folder(dataset_name)
        self.dataset_config = DatasetConfig(self.dataset_folder, output_range=(0.0, 1.0))
        self.dataset_loader = DatasetLoader(self.dataset_config)

        self.base_log_dir = "../logs/AutoEncoding-Anomalies/protocols/{protocol_name}/{dataset_name}" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

    def train_model(self, config: ProtocolTrainConfig):
        log_dir = self.make_log_dir(True)
        if config.initial_epoch > 0:
            self.model.load_weights(self.weights_path.format(config.initial_epoch))

        callbacks = self.make_callback(log_dir, config)

        dataset = self.dataset_loader.train_subset.make_tf_dataset(config.pattern)
        dataset = dataset.batch(config.batch_size).prefetch(-1)

        val_dataset = self.dataset_loader.test_subset.make_tf_dataset(config.pattern)
        val_dataset = val_dataset.batch(config.batch_size)

        self.model.fit(dataset, steps_per_epoch=1000, epochs=config.epochs,
                       validation_data=val_dataset, validation_steps=100,
                       callbacks=callbacks, initial_epoch=config.initial_epoch)

    def make_callback(self,
                      log_dir: str,
                      config: ProtocolTrainConfig
                      ) -> List[Callback]:
        tensorboard = TensorBoard(log_dir=log_dir, update_freq=16, profile_batch=0)
        callbacks = [tensorboard]
        # region Image Callbacks
        if config.image_callbacks_configs is not None:
            for icc in config.image_callbacks_configs:
                callbacks += icc.to_callbacks(tensorboard, self.dataset_loader)
        # endregion
        # region Checkpoint
        model_checkpoint = EagerModelCheckpoint(self.weights_path)
        callbacks.append(model_checkpoint)
        # endregion
        # region Early stopping
        if config.early_stopping_metric is not None:
            early_stopping = EarlyStopping(monitor=config.early_stopping_metric,
                                           mode="min",
                                           restore_best_weights=True,
                                           patience=3
                                           )
            callbacks.append(early_stopping)
        # endregion
        # region AUC
        if config.auc_callbacks_configs is not None:
            for acc in config.auc_callbacks_configs:
                callbacks += [acc.to_callback(tensorboard, self.dataset_loader)]
        # endregion
        return callbacks

    def test_model(self, config: ProtocolTestConfig):
        if config.epoch > 0:
            self.model.load_weights(self.weights_path.format(config.epoch))

        metrics = list(known_metrics.keys())
        anomaly_detector = AnomalyDetector(autoencoder=self.autoencoder,
                                           output_length=config.output_length,
                                           metrics=metrics)

        log_dir = self.make_log_dir(False)
        pattern = config.pattern.with_added_depth().with_added_depth()
        anomaly_detector.predict_and_evaluate(dataset=self.dataset_loader,
                                              pattern=pattern,
                                              log_dir=log_dir,
                                              stride=config.detector_stride,
                                              pre_normalize_predictions=config.pre_normalize_predictions,
                                              additional_config={
                                                  "epoch": config.epoch,
                                                  "model_name": self.model_name,
                                                  **config.kwargs
                                              }
                                              )

    def multi_test_model(self, configs: List[ProtocolTestConfig]):
        for config in configs:
            self.test_model(config)

    def make_log_dir(self, is_train: bool) -> str:
        timestamp = int(time.time())
        sub_folder = "train" if is_train else "anomaly_detection"
        log_dir = os.path.join(self.base_log_dir, sub_folder, "{}".format(timestamp))
        os.makedirs(log_dir)
        save_model_info(self.autoencoder, log_dir)
        return log_dir

    @property
    def weights_path(self) -> str:
        return os.path.join(self.base_log_dir, "weights_{:03d}.hdf5")


def get_dataset_folder(dataset_name) -> str:
    if dataset_name is "ped2":
        return "../datasets/ucsd/ped2"
    elif dataset_name is "ped1":
        return "../datasets/ucsd/ped1"
    elif dataset_name is "emoly":
        return "../datasets/emoly"
    else:
        raise ValueError(dataset_name)
