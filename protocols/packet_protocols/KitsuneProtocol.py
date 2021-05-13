import os
from typing import Union, Dict

from datasets.tfrecord_builders.KitsuneTFRB import KitsuneDataset
from datasets.loaders.kitsune import KitsuneLoader
from protocols import PacketProtocol, ProtocolTestConfig, ProtocolTrainConfig

dataset_id_to_name = {
    KitsuneDataset.ACTIVE_WIRETAP: "Active Wiretap",
    KitsuneDataset.ARP_MITM: "ARP MitM",
    KitsuneDataset.FUZZING: "Fuzzing",
    KitsuneDataset.MIRAI_BOTNET: "Mirai Botnet",
    KitsuneDataset.OS_SCAN: "OS Scan",
    KitsuneDataset.SSDP_FLOOD: "SSDP Flood",
    KitsuneDataset.SSL_RENEGOTIATION: "SSL Renegotiation",
    KitsuneDataset.SYN_DOS: "SYN DoS",
    KitsuneDataset.VIDEO_INJECTION: "Video Injection",
}

dataset_name_to_id = {_name: _id for _id, _name in dataset_id_to_name.items()}

dataset_alias_map = {
    "active_wiretap": "Active Wiretap",
    "arp_mitm": "ARP MitM",
    "fuzzing": "Fuzzing",
    "mirai_botnet": "Mirai Botnet",
    "os_scan": "OS Scan",
    "ssdp_flood": "SSDP Flood",
    "ssl_renegotiation": "SSL Renegotiation",
    "syn_dos": "SYN DoS",
    "video_injection": "Video Injection",
}


class KitsuneProtocol(PacketProtocol):
    def __init__(self,
                 base_log_dir: str,
                 kitsune_dataset: Union[KitsuneDataset, str],
                 epoch=0,
                 config: Dict = None,
                 ):
        dataset_name = KitsuneProtocol.get_dataset_name(kitsune_dataset)
        super(KitsuneProtocol, self).__init__(base_log_dir=base_log_dir,
                                              dataset_name=dataset_name,
                                              epoch=epoch,
                                              config=config)

    def init_dataset_loader(self, dataset_folder, output_range):
        self.dataset_loader = KitsuneLoader(dataset_folder, is_mirai=self.is_mirai,
                                            train_samples_ratio=self.train_samples_ratio)

    @property
    def is_mirai(self) -> bool:
        return self.dataset_name == "Mirai Botnet"

    @property
    def train_samples_ratio(self) -> float:
        return self.get_config_value("train_samples_ratio", default=1.0)

    def make_train_dataset_splits(self, config: ProtocolTrainConfig):
        train_dataset = self.dataset_loader.train_subset.make_tf_dataset(config.pattern,
                                                                         batch_size=config.batch_size,
                                                                         seed=self.seed,  # numpy seed
                                                                         parallel_cores=8)
        if "Benign" in self.dataset_loader.subsets:
            val_dataset = self.dataset_loader.subsets["Benign"].make_tf_dataset(config.pattern,
                                                                                batch_size=config.batch_size,
                                                                                seed=self.seed,  # numpy seed
                                                                                parallel_cores=8)
        else:
            val_dataset = None

        return train_dataset, val_dataset

    def get_config_path(self, protocol_name: str = None, dataset_name: str = None):
        config_path = super(KitsuneProtocol, self).get_config_path(protocol_name=protocol_name,
                                                                   dataset_name=dataset_name)
        if not os.path.exists(config_path):
            config_path = super(KitsuneProtocol, self).get_config_path(protocol_name=protocol_name,
                                                                       dataset_name="Kitsune")
        return config_path

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.epoch,
                                  detector_stride=32,
                                  pre_normalize_predictions=True,
                                  compare_metrics="log_mae")

    @property
    def channels(self) -> int:
        return 115

    @staticmethod
    def get_dataset_name(kitsune_id: Union[KitsuneDataset, str]) -> str:
        if kitsune_id in dataset_name_to_id:
            return kitsune_id
        elif kitsune_id in dataset_id_to_name:
            return dataset_id_to_name[kitsune_id]
        elif kitsune_id in dataset_alias_map:
            return dataset_alias_map[kitsune_id]
        else:
            raise ValueError("Error : {} is not a valid Kitsune dataset ID.".format(kitsune_id))

    @staticmethod
    def get_dataset_id(dataset_name: str) -> KitsuneDataset:
        if dataset_name in dataset_alias_map:
            dataset_name = dataset_alias_map[dataset_name]

        if dataset_name in dataset_name_to_id:
            return dataset_name_to_id[dataset_name]
        else:
            raise ValueError("Error : {} is not a valid Kitsune dataset name.".format(dataset_name))

    @staticmethod
    def is_kitsune_id(kitsune_id: Union[KitsuneDataset, str]) -> bool:
        return (kitsune_id in dataset_name_to_id) or \
               (kitsune_id in dataset_id_to_name) or \
               (kitsune_id in dataset_alias_map)
