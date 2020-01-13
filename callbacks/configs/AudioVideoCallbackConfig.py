from tensorflow.python.keras.callbacks import TensorBoard
from typing import Tuple, Callable

from callbacks import AudioVideoCallback
from callbacks.configs import ModalityCallbackConfig
from datasets import DatasetLoader
from modalities import Pattern, MelSpectrogram


class AudioVideoCallbackConfig(ModalityCallbackConfig):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 is_train_callback: bool,
                 name: str,
                 epoch_freq: int = 1,
                 video_sample_rate: int = 25,
                 audio_sample_rate: int = 48000,
                 mel_spectrogram: MelSpectrogram = None,
                 inputs_are_outputs: bool = True,
                 modality_indices: Tuple[int, int] = None,
                 **kwargs,
                 ):
        super(AudioVideoCallbackConfig, self).__init__(autoencoder=autoencoder,
                                                       pattern=pattern,
                                                       is_train_callback=is_train_callback,
                                                       name=name,
                                                       epoch_freq=epoch_freq,
                                                       inputs_are_outputs=inputs_are_outputs,
                                                       modality_indices=modality_indices,
                                                       **kwargs
                                                       )
        self.video_sample_rate = video_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.mel_spectrogram = mel_spectrogram

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    ) -> AudioVideoCallback:
        image_callbacks = AudioVideoCallback.from_model_and_subset(autoencoder=self.autoencoder,
                                                                   subset=self.get_subset(dataset_loader),
                                                                   pattern=self.pattern,
                                                                   name=self.name,
                                                                   is_train_callback=self.is_train_callback,
                                                                   tensorboard=tensorboard,
                                                                   video_sample_rate=self.video_sample_rate,
                                                                   audio_sample_rate=self.audio_sample_rate,
                                                                   mel_spectrogram=self.mel_spectrogram,
                                                                   epoch_freq=self.epoch_freq,
                                                                   inputs_are_outputs=self.inputs_are_outputs,
                                                                   modality_indices=self.modality_indices,
                                                                   **self.kwargs
                                                                   )
        return image_callbacks
