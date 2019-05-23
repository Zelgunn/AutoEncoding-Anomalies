import tensorflow as tf
import numpy as np
import librosa
from tqdm import tqdm
from typing import Dict, Any

from modalities import Modality


class MelSpectrogram(Modality):
    def __init__(self,
                 window_width: float,
                 window_step: float,
                 mel_filters_count: int):
        """
        :param window_width: Width of the sliding window in seconds.
        :param window_step: Step between the sliding windows in seconds.
        :param mel_filters_count: Number of Mel filters to use.
        """
        super(MelSpectrogram, self).__init__()
        self.window_width = window_width
        self.window_step = window_step
        self.mel_filters_count = mel_filters_count

    def get_config(self) -> Dict[str, Any]:
        base_config = super(MelSpectrogram, self).get_config()
        config = {
            "window_width": self.window_width,
            "window_step": self.window_step,
            "mel_filters_count": self.mel_filters_count
        }
        return {**base_config, **config}

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        return cls.encode_raw(modality_value, np.float32)

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        return cls.decode_raw(parsed_features, tf.float32)

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string),
                cls.shape_id(): cls.tfrecord_shape_parse_function()}

    @classmethod
    def rank(cls) -> int:
        return 2

    def get_output_frame_count(self, input_frame_count: int, frame_rate: int) -> int:
        window_width = int(self.window_width * frame_rate)
        window_step = int(self.window_step * frame_rate)
        return 1 + int((input_frame_count - window_width) / window_step)

    def wave_to_mel_spectrogram(self, frames: np.ndarray, frame_rate: int):
        window_width = int(self.window_width * frame_rate)
        window_step = int(self.window_step * frame_rate)
        return wave_to_mel_spectrogram(frames=frames,
                                       frame_rate=frame_rate,
                                       nfft=window_width,
                                       hop_length=window_step,
                                       mel_filters_count=self.mel_filters_count,
                                       to_db=False)


def wave_to_mel_spectrogram(frames: np.ndarray,
                            frame_rate: int,
                            nfft: int,
                            hop_length: int,
                            mel_filters_count: int,
                            to_db=False,
                            ) -> np.ndarray:
    if 1 in frames.shape:
        frames = frames.squeeze()

    if frames.ndim not in [1, 2]:
        raise ValueError("Frames must either have 1 or 2 dimensions, "
                         "got {} with shape {}".format(frames.ndim, frames.shape))

    if frames.ndim == 2:
        frames = frames.mean(axis=1)

    features = librosa.feature.melspectrogram(frames, sr=frame_rate, n_mels=mel_filters_count,
                                              hop_length=hop_length, n_fft=nfft)

    if to_db:
        features = librosa.power_to_db(features, ref=np.max)

    features = np.transpose(features)

    return features


def mel_spectrogram_to_wave(features: np.ndarray,
                            frame_rate: int,
                            nfft: int,
                            hop_length: int,
                            mel_filters_count: int,
                            from_db=False,
                            iterations=100
                            ) -> np.ndarray:
    features = np.transpose(features)

    if from_db:
        features = librosa.db_to_power(features)

    # region Revert Mel scale
    filters = librosa.filters.mel(sr=frame_rate, n_fft=nfft, n_mels=mel_filters_count)
    filters = np.transpose(filters)
    features = np.dot(filters, features)
    # endregion

    # region Iterative inversion
    x = None
    phase = 2 * np.pi * np.random.random_sample(features.shape) - np.pi
    for _ in tqdm(range(iterations)):
        s = features * np.exp(1j * phase)
        x = librosa.istft(s, hop_length)
        phase = np.angle(librosa.stft(x, nfft))
    # endregion

    return x
