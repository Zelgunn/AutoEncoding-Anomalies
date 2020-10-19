import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, TensorBoard, TerminateOnNaN, ModelCheckpoint
from tensorboard.plugins import projector
import time
import os
from typing import List, Callable, Dict, Sequence, Union

from anomaly_detection import AnomalyDetector, known_metrics
from callbacks.configs import ModalityCallbackConfig, AUCCallbackConfig, AnomalyDetectorCallbackConfig
from callbacks import CustomModelCheckpoint
from datasets import DatasetLoader, SingleSetConfig
from misc_utils.train_utils import save_model_info
from modalities import Pattern
from custom_tf_models import LED


class ProtocolTrainConfig(object):
    def __init__(self,
                 batch_size: int,
                 pattern: Pattern,
                 steps_per_epoch: int,
                 epochs: int,
                 initial_epoch: int,
                 validation_steps: int,
                 save_frequency: Union[str, int],
                 modality_callback_configs: List[ModalityCallbackConfig] = None,
                 auc_callback_configs: List[AUCCallbackConfig] = None,
                 anomaly_detector_callback_configs: List[AnomalyDetectorCallbackConfig] = None,
                 ):
        self.batch_size = batch_size
        self.pattern = pattern
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.validation_steps = validation_steps
        self.save_frequency = save_frequency
        self.modality_callback_configs = modality_callback_configs
        self.auc_callback_configs = auc_callback_configs
        self.anomaly_detector_callback_configs = anomaly_detector_callback_configs


class ProtocolTestConfig(object):
    def __init__(self,
                 pattern: Pattern,
                 epoch: int,
                 detector_stride: int,
                 pre_normalize_predictions: bool,
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 **kwargs,
                 ):
        self.pattern = pattern
        self.epoch = epoch
        self.detector_stride = detector_stride
        self.pre_normalize_predictions = pre_normalize_predictions
        self.additional_metrics = additional_metrics
        self.kwargs = kwargs


class Protocol(object):
    def __init__(self,
                 model: Model,
                 dataset_name: str,
                 protocol_name: str,
                 autoencoder: Callable = None,
                 model_name: str = None,
                 output_range=(0.0, 1.0),
                 seed=None
                 ):
        self.model = model
        if autoencoder is None:
            autoencoder = model
        self.autoencoder = autoencoder
        if model_name is None:
            model_name = model.name
        self.model_name = model_name
        self.autoencoder_name = autoencoder.name if hasattr(autoencoder, "name") else autoencoder.__name__
        self.protocol_name = protocol_name

        self.dataset_name = dataset_name
        self.dataset_folder = get_dataset_folder(dataset_name)
        self.dataset_config = SingleSetConfig(self.dataset_folder, output_range=output_range)
        self.dataset_loader = DatasetLoader(self.dataset_config)

        self.base_log_dir = "../logs/AEA/{protocol_name}/{dataset_name}" \
            .format(protocol_name=protocol_name, dataset_name=dataset_name)

        self.seed = seed

    def train_model(self, config: ProtocolTrainConfig, **kwargs):
        print("Protocol - Training : Loading weights ...")
        self.load_weights(epoch=config.initial_epoch)

        print("Protocol - Training : Making log dirs ...")
        log_dir = self.make_log_dir("train")

        print("Protocol - Training : Making callbacks ...")
        callbacks = self.make_callback(log_dir, config)

        print("Protocol - Training : Making datasets ...")
        subset = self.dataset_loader.train_subset
        train_dataset, val_dataset = subset.make_tf_datasets_splits(config.pattern,
                                                                    split=0.9,
                                                                    batch_size=config.batch_size,
                                                                    seed=self.seed,
                                                                    parallel_cores=8)

        print("Protocol - Training : Fit loop ...")
        self.model.fit(train_dataset, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                       validation_data=val_dataset, validation_steps=config.validation_steps,
                       callbacks=callbacks, initial_epoch=config.initial_epoch, **kwargs)

    def make_callback(self,
                      log_dir: str,
                      config: ProtocolTrainConfig
                      ) -> List[Callback]:
        print("Protocol - Make Callbacks - Tensorboard ...")
        tensorboard = TensorBoard(log_dir=log_dir, update_freq=32, profile_batch=0,
                                  histogram_freq=1, write_images=False)
        callbacks = [tensorboard, TerminateOnNaN()]
        # region Checkpoint
        print("Protocol - Make Callbacks - Checkpoint ...")
        weights_path = os.path.join(log_dir, "weights_{epoch:03d}")
        model_checkpoint = ModelCheckpoint(filepath=weights_path, save_freq=config.save_frequency)
        callbacks.append(model_checkpoint)
        # callbacks.append(CustomModelCheckpoint(filepath=os.path.join(log_dir, "full_checkpoint_{epoch:03d}"),
        #                                        save_weights_only=False))
        # endregion
        # region AUC
        if config.auc_callback_configs is not None:
            print("Protocol - Make Callbacks - AUC callbacks ...")
            for acc in config.auc_callback_configs:
                print("Protocol - Make AUC Callbacks - {} callback ...".format(acc.prefix))
                callback = acc.to_callback(tensorboard, self.dataset_loader, self.seed)
                callbacks.append(callback)
        # endregion
        # region Anomaly detector (full test)
        if config.anomaly_detector_callback_configs is not None:
            print("Protocol - Make Callbacks - Anomaly detector callbacks")
            for adc in config.anomaly_detector_callback_configs:
                callback = adc.to_callback(tensorboard, self.dataset_loader)
                callbacks.append(callback)
        # endregion
        # region Modality Callbacks
        if config.modality_callback_configs is not None:
            print("Protocol - Make Callbacks - Modality callbacks")
            for mcc in config.modality_callback_configs:
                print("Protocol - Make Modality Callbacks - {} callback ...".format(mcc.name))
                callback = mcc.to_callback(tensorboard, self.dataset_loader, self.seed)
                callbacks.append(callback)
        # endregion
        # region Early stopping
        # print("Protocol - Make Callbacks - Early Stopping ...")
        # if config.early_stopping_metric is not None:
        #     early_stopping = EarlyStopping(monitor=config.early_stopping_metric,
        #                                    mode="min",
        #                                    patience=5
        #                                    )
        #     callbacks.append(early_stopping)
        # endregion
        return callbacks

    def test_model(self, config: ProtocolTestConfig):
        self.load_weights(epoch=config.epoch)

        compare_metrics = list(known_metrics.keys())
        additional_metrics = self.additional_test_metrics
        if config.additional_metrics is not None:
            additional_metrics = [*additional_metrics, *config.additional_metrics]

        anomaly_detector = AnomalyDetector(autoencoder=self.autoencoder,
                                           pattern=config.pattern,
                                           compare_metrics=compare_metrics,
                                           additional_metrics=additional_metrics)

        log_dir = self.make_log_dir("anomaly_detection")

        if self.dataset_name is "emoly":
            folders = self.dataset_loader.test_subset.subset_folders
            folders = [folder for folder in folders if "acted" in folder]
            self.dataset_loader.test_subset.subset_folders = folders

        anomaly_detector.predict_and_evaluate(dataset=self.dataset_loader,
                                              log_dir=log_dir,
                                              stride=config.detector_stride,
                                              pre_normalize_predictions=config.pre_normalize_predictions,
                                              additional_config={
                                                  "epoch": config.epoch,
                                                  "model_name": self.autoencoder_name,
                                                  **config.kwargs
                                              }
                                              )

    @property
    def additional_test_metrics(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return getattr(self.model, "additional_test_metrics", [])

    def log_model_latent_codes(self, config: ProtocolTestConfig):
        self.load_weights(epoch=config.epoch)

        anomaly_detector = AnomalyDetector(autoencoder=self.autoencoder,
                                           pattern=config.pattern,
                                           compare_metrics=[],
                                           additional_metrics=None)

        log_dir = self.make_log_dir("latent_codes")

        latent_codes, samples_infos = anomaly_detector.compute_latent_codes_on_dataset(dataset=self.dataset_loader,
                                                                                       stride=4)

        self.log_latent_codes_metadata(samples_infos, log_dir)

        embeddings = tf.Variable(initial_value=latent_codes)
        checkpoint = tf.train.Checkpoint(embedding=embeddings)
        checkpoint.save(os.path.join(log_dir, "embeddings.ckpt"))

        projector_config = projector.ProjectorConfig()
        config_embeddings = projector_config.embeddings.add()
        config_embeddings.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        # config_embeddings.tensor_name = "{}/latent_codes".format(self.dataset_name)
        config_embeddings.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, projector_config)

    @staticmethod
    def log_latent_codes_metadata(samples_infos: Dict[str, Sequence], log_dir: str):
        info_names = list(samples_infos.keys())
        info_values = [samples_infos[info_name] for info_name in info_names]

        info_count = len(info_names)
        values_count = len(info_values[0])

        with open(os.path.join(log_dir, "metadata.tsv"), "w") as metadata_file:
            for info_index in range(info_count):
                space = "\t" if ((info_index + 1) != info_count) else "\n"
                header = info_names[info_index]
                metadata_file.write("{}{}".format(header, space))

            for value_index in range(values_count):
                for info_index in range(info_count):
                    space = "\t" if ((info_index + 1) != info_count) else "\n"
                    value = info_values[info_index][value_index]
                    metadata_file.write("{}{}".format(value, space))

    def make_log_dir(self, sub_folder: str) -> str:
        timestamp = str(int(time.time()))
        log_dir = os.path.join(self.base_log_dir, sub_folder, self.model_name, timestamp)
        os.makedirs(log_dir)
        save_model_info(self.model, log_dir)
        return log_dir

    def load_weights(self, epoch: int):
        if epoch > 0:
            weights_path = os.path.join(self.base_log_dir, "weights_{epoch:03d}")
            weights_path = weights_path.format(epoch=epoch)
            self.model.load_weights(weights_path)


def get_dataset_folder(dataset_name: str) -> str:
    known_datasets = {
        "ped2": "../datasets/ucsd/ped2",
        "ped1": "../datasets/ucsd/ped1",

        "subway_exit": "../datasets/subway/exit",
        "subway_entrance": "../datasets/subway/entrance",
        "subway_mall1": "../datasets/subway/mall3",
        "subway_mall2": "../datasets/subway/mall3",
        "subway_mall3": "../datasets/subway/mall3",

        "shanghaitech": "../datasets/shanghaitech",
        "avenue": "../datasets/avenue",

        "emoly": "../datasets/emoly",
        "audioset": "../datasets/audioset",
    }

    if dataset_name in known_datasets:
        return known_datasets[dataset_name]
    else:
        raise ValueError("Unknown dataset : {}".format(dataset_name))
