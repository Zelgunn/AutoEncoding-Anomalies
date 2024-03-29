import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, TensorBoard, TerminateOnNaN, ModelCheckpoint
from tensorboard.plugins import projector
import time
import os
from typing import List, Callable, Dict, Sequence, Union, Optional

from custom_tf_models import AE
from anomaly_detection import AnomalyDetector
from callbacks.configs import ModalityCallbackConfig, AUCCallbackConfig, AnomalyDetectorCallbackConfig
from datasets import DatasetLoader, SingleSetConfig, TFRecordDatasetLoader
from misc_utils.train_utils import save_model_info
from modalities import Pattern


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
                 compare_metrics: Union[Union[str, Callable], List[Union[str, Callable]]],
                 additional_metrics: List[Callable[[tf.Tensor], tf.Tensor]] = None,
                 **kwargs,
                 ):
        self.pattern = pattern
        self.epoch = epoch
        self.detector_stride = detector_stride
        self.pre_normalize_predictions = pre_normalize_predictions
        compare_metrics = compare_metrics if isinstance(compare_metrics, list) else [compare_metrics]
        self.compare_metrics: List[Union[str, Callable]] = compare_metrics
        self.additional_metrics = additional_metrics
        self.kwargs = kwargs


class Protocol(object):
    def __init__(self,
                 dataset_name: str,
                 protocol_name: str,
                 base_log_dir: str,
                 model: Optional[Model],
                 output_range=(0.0, 1.0),
                 seed=None
                 ):
        self.seed = seed
        tf.random.set_seed(seed)

        if model is None:
            model = self.make_model()

        self.model = model
        self.protocol_name = protocol_name
        self.dataset_name = dataset_name
        self.base_log_dir = base_log_dir

        self.dataset_folder = get_dataset_folder(dataset_name)
        self.dataset_config: Optional[SingleSetConfig] = None
        self.dataset_loader: Optional[DatasetLoader] = None
        self.init_dataset_loader(self.dataset_folder, output_range)

    def init_dataset_loader(self, dataset_folder, output_range):
        self.dataset_config = SingleSetConfig(dataset_folder, output_range=output_range)
        self.dataset_loader = TFRecordDatasetLoader(self.dataset_config)

    def make_model(self) -> Model:
        raise NotImplementedError

    def make_train_dataset_splits(self, config: ProtocolTrainConfig):
        split_folders = {}
        subset = self.dataset_loader.train_subset
        split = 0.9
        if split >= 1.0:
            train_dataset = subset.make_tf_dataset(config.pattern,
                                                   batch_size=config.batch_size,
                                                   seed=self.seed,
                                                   parallel_cores=8)
            val_dataset = None
        else:
            train_dataset, val_dataset = subset.make_tf_datasets_splits(config.pattern,
                                                                        split=split,
                                                                        batch_size=config.batch_size,
                                                                        seed=self.seed,  # numpy seed
                                                                        parallel_cores=8,
                                                                        split_folders=split_folders)

        for dataset_name in split_folders:
            print("Elements in {} :".format(dataset_name))
            for path in split_folders[dataset_name]:
                print("===> {}".format(os.path.basename(path)))

        return train_dataset, val_dataset

    def train_model(self, config: ProtocolTrainConfig, **kwargs):
        print("Protocol - Training : Starting training on {}.".format(self.dataset_name))
        self.load_weights(epoch=config.initial_epoch, expect_partial=False)

        print("Protocol - Training : Making log dirs ...")
        log_dir = self.make_log_dir("train")

        print("Protocol - Training : Making callbacks ...")
        callbacks = self.make_callback(log_dir, config)

        print("Protocol - Training : Making datasets ...")
        train_dataset, val_dataset = self.make_train_dataset_splits(config=config)

        self.model.summary()
        exit()

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
                                  histogram_freq=0, write_images=False)
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
        self.load_weights(epoch=config.epoch, expect_partial=True)

        compare_metrics = config.compare_metrics if isinstance(self.model, AE) else None

        additional_metrics = self.additional_test_metrics
        if config.additional_metrics is not None:
            additional_metrics = [*additional_metrics, *config.additional_metrics]

        anomaly_detector = AnomalyDetector(model=self.autoencoder,
                                           pattern=config.pattern,
                                           compare_metrics=compare_metrics,
                                           additional_metrics=additional_metrics)

        log_dir = self.make_log_dir("anomaly_detection")

        if self.dataset_name == "emoly":
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

    def log_model_latent_codes(self, config: ProtocolTestConfig):
        self.load_weights(epoch=config.epoch, expect_partial=True)

        anomaly_detector = AnomalyDetector(model=self.autoencoder,
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
        log_dir = os.path.join(self.dataset_log_dir, sub_folder, self.model_name, timestamp)
        os.makedirs(log_dir)
        save_model_info(self.model, log_dir)
        return log_dir

    # noinspection PyUnusedLocal
    def load_weights(self, epoch: int, verbose=False, expect_partial=False):
        weights_path = os.path.join(self.dataset_log_dir, "weights_{epoch:03d}")
        if verbose:
            print("Protocol : Loading weights from {} ..".format(weights_path))

        if epoch > 0:
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(weights_path.format(epoch=epoch))
            # if expect_partial:
            #     checkpoint.expect_partial()

    # region Properties
    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def autoencoder(self) -> Callable:
        return self.model

    @property
    def autoencoder_name(self) -> str:
        if hasattr(self.autoencoder, "name"):
            return self.autoencoder.name
        else:
            return self.autoencoder.__name__

    @property
    def dataset_log_dir(self) -> str:
        return os.path.join(self.base_log_dir, self.protocol_name, self.dataset_name)

    @property
    def additional_test_metrics(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return getattr(self.model, "additional_test_metrics", [])

    # endregion


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

        "Active Wiretap": "../datasets/kitsune/Active Wiretap",
        "ARP MitM": "../datasets/kitsune/ARP MitM",
        "Fuzzing": "../datasets/kitsune/Fuzzing",
        "Mirai Botnet": "../datasets/kitsune/Mirai Botnet",
        "OS Scan": "../datasets/kitsune/OS Scan",
        "SSDP Flood": "../datasets/kitsune/SSDP Flood",
        "SSL Renegotiation": "../datasets/kitsune/SSL Renegotiation",
        "SYN DoS": "../datasets/kitsune/SYN DoS",
        "Video Injection": "../datasets/kitsune/Video Injection",

        "CIDDS.TCP": "../datasets/cidds/OpenStack",
        "CIDDS.UDP": "../datasets/cidds/OpenStack",
    }

    if dataset_name in known_datasets:
        return known_datasets[dataset_name]
    else:
        raise ValueError("Unknown dataset : {}".format(dataset_name))
