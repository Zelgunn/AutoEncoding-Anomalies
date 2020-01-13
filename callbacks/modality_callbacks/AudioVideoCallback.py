import tensorflow as tf
from tensorflow import keras
from typing import List, Union

from callbacks import ModalityCallback
from callbacks.modality_callbacks.AudioCallback import audio_sample_summary
from misc_utils.summary_utils import image_summary as video_sample_summary, convert_tensor_uint8
from modalities import MelSpectrogram


class AudioVideoCallback(ModalityCallback):
    def __init__(self,
                 inputs: List[tf.Tensor],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: List[tf.Tensor] = None,
                 logged_output_indices=(0, 1),
                 name: str = "AudioVideoCallback",
                 max_outputs: int = 3,
                 video_sample_rate=25,
                 audio_sample_rate=48000,
                 mel_spectrogram: MelSpectrogram = None,
                 **kwargs):
        super(AudioVideoCallback, self).__init__(inputs=inputs, model=model,
                                                 tensorboard=tensorboard, is_train_callback=is_train_callback,
                                                 update_freq=update_freq, epoch_freq=epoch_freq,
                                                 outputs=outputs, logged_output_indices=logged_output_indices,
                                                 name=name, max_outputs=max_outputs, **kwargs)
        self.video_sample_rate = video_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.mel_spectrogram = mel_spectrogram

    def write_model_summary(self, step: int):
        pred_outputs = self.summary_model(self.inputs)
        pred_outputs = self.extract_logged_modalities(pred_outputs)
        self.samples_summary(data=pred_outputs, step=step, suffix="predicted")

    def samples_summary(self, data: List[tf.Tensor], step: int, suffix: str):
        for sample in data:
            self.sample_summary(data=sample, step=step, suffix=suffix)

    def sample_summary(self, data: tf.Tensor, step: int, suffix: str):
        name = "{}_{}".format(self.name, suffix)
        if data.shape.rank >= 5:
            data = convert_tensor_uint8(data)
            video_sample_summary(data=data,
                                 fps=self.video_sample_rate,
                                 step=step,
                                 name=name,
                                 max_outputs=self.max_outputs
                                 )
        else:
            audio_sample_summary(data=data,
                                 sample_rate=self.audio_sample_rate,
                                 step=step,
                                 name=name,
                                 mel_spectrogram=self.mel_spectrogram,
                                 max_outputs=self.max_outputs)
