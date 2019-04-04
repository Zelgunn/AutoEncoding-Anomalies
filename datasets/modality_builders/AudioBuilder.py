from typing import Union, List, Dict, Any

from datasets.modality_builders import ModalityBuilder, AudioReader


class AudioBuilder(ModalityBuilder):
    def __init__(self,
                 modalities: Union[str, List[str], Dict[str, Dict[str, Any]]],
                 audio_reader: AudioReader):
        super(AudioBuilder, self).__init__(modalities=modalities)

        self.audio_reader = audio_reader

    @classmethod
    def supported_modalities(cls):
        return ["raw_audio", "mfcc"]

    def __iter__(self):
        raise NotImplementedError

    def get_buffer_shape(self, modality: str):
        raise NotImplementedError
