import numpy as np
from enum import IntEnum
from typing import Union, List


class AudioReaderMode(IntEnum):
    AUDIO_FILE = 0,
    NP_ARRAY = 1,


class AudioReader(object):
    def __init__(self,
                 audio_source: Union[str, np.ndarray, List[str]],
                 mode: AudioReaderMode = None):
        self.audio_source = audio_source
        self.mode = mode
