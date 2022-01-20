import os
from typing import Union, Dict

from protocols import PacketProtocol, ProtocolTestConfig, ProtocolTrainConfig
from datasets.loaders.nprecords.nid_loaders.CIDDSLoader import CIDDSLoader, CIDDSNetworkProtocol
from datasets.data_readers.packet_readers.CIDDSPacketReader import guess_network_protocol


class CIDDSProtocol(PacketProtocol):
    def __init__(self,
                 base_log_dir: str,
                 network_protocol: Union[CIDDSNetworkProtocol, str],
                 epoch=0,
                 config: Dict = None,
                 ):
        self.network_protocol = guess_network_protocol(network_protocol)
        dataset_name = "CIDDS.{}".format(self.network_protocol.name)
        super(CIDDSProtocol, self).__init__(base_log_dir=base_log_dir,
                                            dataset_name=dataset_name,
                                            epoch=epoch,
                                            config=config)

    def init_dataset_loader(self, dataset_folder, output_range):
        self.dataset_loader = CIDDSLoader(dataset_folder, protocol=self.network_protocol,
                                          min_normal_segment_length=self.input_length,
                                          train_samples_ratio=0.8)

    def get_config_path(self, protocol_name: str = None, dataset_name: str = None):
        config_path = super(CIDDSProtocol, self).get_config_path(protocol_name=protocol_name,
                                                                 dataset_name=dataset_name)
        if not os.path.exists(config_path):
            config_path = super(CIDDSProtocol, self).get_config_path(protocol_name=protocol_name,
                                                                     dataset_name="CIDDS")
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
        return 23
