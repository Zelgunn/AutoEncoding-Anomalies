import tensorflow as tf
from tensorflow import keras
from typing import List, Union

from callbacks import ModalityCallback
from modalities import MelSpectrogram


class AudioCallback(ModalityCallback):
    def __init__(self,
                 inputs: Union[tf.Tensor, List[tf.Tensor]],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: Union[tf.Tensor, List[tf.Tensor]] = None,
                 logged_output_indices=0,
                 name: str = "AudioCallback",
                 max_outputs: int = 3,
                 sample_rate=48000,
                 mel_spectrogram: MelSpectrogram = None,
                 **kwargs):
        super(AudioCallback, self).__init__(inputs=inputs, model=model,
                                            tensorboard=tensorboard, is_train_callback=is_train_callback,
                                            update_freq=update_freq, epoch_freq=epoch_freq,
                                            outputs=outputs, logged_output_indices=logged_output_indices,
                                            name=name, max_outputs=max_outputs, **kwargs)
        self.mel_spectrogram = mel_spectrogram
        self.sample_rate = sample_rate

        check_audio_rank(self.true_outputs)

    def write_model_summary(self, step: int):
        pred_outputs = self.summary_model(self.inputs)
        pred_outputs = self.extract_logged_modalities(pred_outputs)
        self.samples_summary(data=pred_outputs, step=step, suffix="predicted")

    def sample_summary(self, data: tf.Tensor, step: int, suffix: str, **kwargs):
        audio_sample_summary(data=data,
                             sample_rate=self.sample_rate,
                             step=step,
                             name="{}_{}".format(self.name, suffix),
                             mel_spectrogram=self.mel_spectrogram,
                             max_outputs=self.max_outputs)


def audio_sample_summary(data: tf.Tensor,
                         sample_rate: int,
                         step: int,
                         name: str,
                         mel_spectrogram: MelSpectrogram = None,
                         max_outputs: int = 3,
                         ):
    if mel_spectrogram is not None:
        if isinstance(data, tf.Tensor):
            data = data.numpy()
        data = (data - 1.0) * 80.0
        data = mel_spectrogram.mel_spectrograms_to_wave(data, sample_rate)

    if len(data.shape) == 2:
        data = tf.expand_dims(data, axis=-1)
    data /= tf.reduce_max(tf.abs(data), axis=(1, 2), keepdims=True)

    tf.summary.audio(name=name, data=data, sample_rate=sample_rate, step=step, max_outputs=max_outputs)


def check_audio_rank(data: Union[tf.Tensor, List[tf.Tensor]]):
    if isinstance(data, list) or isinstance(data, tuple):
        for sample in data:
            check_audio_rank(sample)

    elif data.shape.rank != 3:
        raise ValueError("Incorrect rank for images/video, expected rank == 3, got {} with rank {}."
                         .format(data.shape, data.shape.rank))
