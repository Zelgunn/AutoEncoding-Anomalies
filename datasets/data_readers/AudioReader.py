import numpy as np
from moviepy.editor import AudioFileClip
from pydub.utils import mediainfo
from enum import IntEnum
from typing import Union, List, Any, Iterator


def check_int_type(value: Any, name: str):
    if value is not None and not isinstance(value, int):
        raise ValueError("`{}` must be of type `int` when provided, received '{}'.".format(name, type(value)))


class AudioReaderMode(IntEnum):
    AUDIO_FILE = 0,
    NP_ARRAY = 1,


class AudioReader(object):
    def __init__(self,
                 audio_source: Union[str, np.ndarray, List[str]],
                 mode: AudioReaderMode = None,
                 frequency: int = None,
                 start=None, end=None):

        check_int_type(frequency, "frequency")
        check_int_type(start, "start")
        check_int_type(end, "end")

        if mode is None:
            if isinstance(audio_source, np.ndarray):
                mode = AudioReaderMode.NP_ARRAY
                if audio_source.ndim not in [1, 2]:
                    raise ValueError("Rank of `audio_source` must either be 1 or 2, got {}".format(audio_source.ndim))

            elif isinstance(audio_source, str):
                mode = AudioReaderMode.AUDIO_FILE
                if frequency is None:
                    frequency = int(mediainfo(audio_source)["sample_rate"])
                audio_source = AudioFileClip(audio_source, fps=frequency)

            elif isinstance(audio_source, AudioFileClip):
                mode = AudioReaderMode.AUDIO_FILE
                if frequency is None:
                    frequency = audio_source.fps

            else:
                raise ValueError("`audio_source` of type '{}' is not supported.".format(type(audio_source)))

        self.mode: AudioReaderMode = mode
        self.audio_source: Union[AudioFileClip, np.ndarray] = audio_source

        if isinstance(audio_source, np.ndarray) and frequency is None:
            raise ValueError("You must provide a value for `frequency` when providing a numpy array.")

        self.frequency: int = frequency

        # region End / Start
        if self.mode == AudioReaderMode.AUDIO_FILE:
            max_frame_count = int(self.frequency * self.audio_source.duration)
        else:
            max_frame_count = self.audio_source.shape[0]

        if end is None:
            self.end = max_frame_count
        elif end < 0:
            self.end = max_frame_count + end
        else:
            self.end = min(end, max_frame_count)

        if start is None:
            self.start = 0
        elif start < 0:
            self.start = max_frame_count + start
        elif start < max_frame_count:
            self.start = start
        else:
            raise ValueError("`start` is after the end of the stream "
                             "{}(start) >= {}(frame_count)".format(start, max_frame_count))

        if self.end <= self.start:
            raise ValueError("`start` must be less than `end`, got {}(start) and {}(end)".format(self.start, self.end))
        # endregion

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.mode == AudioReaderMode.AUDIO_FILE:
            iterator = self.audio_source.iter_frames(dtype=np.float32)

            for i, frame in enumerate(iterator):
                if i >= self.start:
                    yield frame

                if (i + 1) == self.end:
                    break
        else:
            for i in range(self.start, self.end):
                yield self.audio_source[i]

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def channels_count(self) -> int:
        if self.mode == AudioReaderMode.AUDIO_FILE:
            return self.audio_source.nchannels
        elif self.audio_source.ndim == 1:
            return 1
        else:
            return self.audio_source.shape[1]
