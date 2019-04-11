from typing import Dict, Type, List, Optional, Any

from modalities import Modality


class ModalityCollection(object):
    def __init__(self,
                 modalities: Optional[List[Modality]] = None
                 ):
        self.modalities: Dict[Type[Modality], Modality] = {}
        if modalities is not None:
            for modality in modalities:
                self.modalities[type(modality)] = modality

    def __iter__(self):
        for modality in self.modalities.values():
            yield modality

    def __contains__(self, item):
        if isinstance(item, Modality):
            return item in self.modalities.values()
        else:
            return item in self.modalities

    def __getitem__(self, item):
        return self.modalities[item]

    def __len__(self):
        return len(self.modalities)

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        config = {}
        for modality in self.modalities.values():
            config[modality.tfrecord_id()] = modality.get_config()
        return config
