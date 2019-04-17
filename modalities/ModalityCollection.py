from typing import Dict, Type, List, Optional, Any

from modalities import Modality, RawVideo, OpticalFlow, DoG, RawAudio, MFCCs, ModalityShape

EXISTING_MODALITIES: List[Type[Modality]] = [RawVideo, OpticalFlow, DoG, RawAudio, MFCCs]
MODALITY_ID_TO_CLASS: Dict[str, Type[Modality]] = {modality_type.id(): modality_type
                                                   for modality_type in EXISTING_MODALITIES}


class ModalityCollection(object):

    def __init__(self,
                 modalities: Optional[List[Modality]] = None
                 ):
        self._modalities: Dict[Type[Modality], Modality] = {}
        if modalities is not None:
            for modality in modalities:
                self._modalities[type(modality)] = modality

    def __iter__(self):
        for modality in self._modalities.values():
            yield modality

    def __contains__(self, item):
        if isinstance(item, Modality):
            return item in self._modalities.values()
        else:
            return item in self._modalities

    def __getitem__(self, item):
        return self._modalities[item]

    def __len__(self):
        return len(self._modalities)

    def types(self):
        return self._modalities.keys()

    def items(self):
        return self._modalities.items()

    def ids(self):
        for modality_type in self._modalities:
            yield modality_type.id()

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        config = {}
        for modality in self._modalities.values():
            config[modality.id()] = modality.get_config()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Dict[str, Any]]):
        modalities: List[Modality] = []
        for modality_id, modality_config in config.items():
            modality_type = MODALITY_ID_TO_CLASS[modality_id]
            modality = modality_type.from_config(modality_config)
            modalities.append(modality)
        return cls(modalities)

    def set_modalities_shapes(self,
                              modalities_shapes: Dict[Type[Modality], ModalityShape],
                              filter_missing_modalities=False):
        if filter_missing_modalities:
            self.filter(list(modalities_shapes.keys()))

        for modality_type, modality_shape in modalities_shapes.items():
            self._modalities[modality_type].io_shape = modality_shape

    def filter(self, modalities_types: List[Type[Modality]]):
        keys = list(self._modalities.keys())
        for key in keys:
            if key not in modalities_types:
                self._modalities.pop(key)

    def get_tfrecord_features(self) -> Dict[str, tuple]:
        features = {}
        for modality in self._modalities:
            features.update(modality.tfrecord_features())
        return features

    @staticmethod
    def modality_id_to_class(modality_id: str) -> Type[Modality]:
        return MODALITY_ID_TO_CLASS[modality_id]
