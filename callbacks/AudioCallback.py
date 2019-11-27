import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import def_function
from typing import List, Union

from callbacks import TensorBoardPlugin
from datasets import SubsetLoader
from modalities import Pattern, MelSpectrogram


class AudioCallback(TensorBoardPlugin):
    def __init__(self,
                 summary_function: def_function.Function,
                 summary_inputs,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 is_one_shot=False):
        super(AudioCallback, self).__init__(tensorboard, update_freq, epoch_freq)
        self.summary_function = summary_function
        self.summary_inputs = summary_inputs
        self.writer_name = self.train_run_name if is_train_callback else self.validation_run_name
        self.is_one_shot = is_one_shot

    def _write_logs(self, index):
        if self.is_one_shot and self.summary_function is None:
            return

        with self._get_writer(self.writer_name).as_default():
            if self.summary_function is not None:
                self.summary_function(self.summary_inputs, step=index)

        if self.is_one_shot:
            self.summary_function = None

    @staticmethod
    def from_model_and_subset(autoencoder: keras.Model,
                              subset: Union[SubsetLoader],
                              pattern: Pattern,
                              name: str,
                              is_train_callback: bool,
                              tensorboard: keras.callbacks.TensorBoard,
                              update_freq="epoch",
                              epoch_freq=1,
                              mel_spectrogram: MelSpectrogram = None,
                              sample_rate=48000,
                              inputs_are_outputs=True,
                              modality_index=None,
                              ) -> List["AudioCallback"]:
        batch = subset.get_batch(batch_size=4, pattern=pattern)

        if inputs_are_outputs:
            inputs = outputs = batch
        else:
            inputs, outputs = batch

        if modality_index is None:
            audio_outputs = outputs
        else:
            audio_outputs = outputs[modality_index]

        def to_wave(data):
            if mel_spectrogram is not None:
                if isinstance(data, tf.Tensor):
                    data = data.numpy()
                data = (data - 1) * 80
                data = mel_spectrogram.mel_spectrograms_to_wave(data, sample_rate)
            if len(data.shape) == 2:
                data = tf.expand_dims(data, axis=-1)
            data *= 1.0 / tf.reduce_max(tf.abs(data), axis=(1, 2))
            return data

        audio_outputs = to_wave(audio_outputs)

        def one_shot_function(data, step):
            return tf.summary.audio(name="{}_true".format(name), data=data, sample_rate=sample_rate,
                                    step=step, max_outputs=4)

        def repeated_function(data, step):
            data = autoencoder(data)
            if modality_index is not None:
                data = data[modality_index]
            data = to_wave(data)
            return tf.summary.audio(name="{}_pred".format(name), data=data, sample_rate=sample_rate,
                                    step=step, max_outputs=4)

        one_shot_callback = AudioCallback(summary_function=one_shot_function, summary_inputs=audio_outputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=True)

        repeated_callback = AudioCallback(summary_function=repeated_function, summary_inputs=inputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=False)

        return [one_shot_callback, repeated_callback]
