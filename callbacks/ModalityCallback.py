import tensorflow as tf
from tensorflow import keras
from typing import List, Union, Callable
from abc import ABC, abstractmethod

from callbacks import TensorBoardPlugin
from misc_utils.general import to_list
from datasets import SubsetLoader
from modalities import Pattern


class ModalityCallback(TensorBoardPlugin, ABC):
    def __init__(self,
                 inputs: Union[tf.Tensor, List[tf.Tensor]],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: Union[tf.Tensor, List[tf.Tensor]] = None,
                 logged_output_indices=0,
                 name: str = "ModalityCallback",
                 max_outputs: int = 4,
                 **kwargs
                 ):
        super(ModalityCallback, self).__init__(tensorboard, update_freq, epoch_freq)
        self.__dict__.update(kwargs)

        self.inputs = inputs
        self.outputs = inputs if outputs is None else outputs
        self.summary_model = model
        self.writer_name = self.train_run_name if is_train_callback else self.validation_run_name

        self.true_outputs_were_logged = False
        self.logged_output_indices = to_list(logged_output_indices)

        self.name = name
        self.max_outputs = max_outputs

        self.true_outputs = self.extract_logged_modalities(self.outputs)

    def _write_logs(self, index):
        with self._get_writer(self.writer_name).as_default():
            if not self.true_outputs_were_logged:
                self.samples_summary(self.true_outputs, step=index, suffix="true")
                self.true_outputs_were_logged = True

            self.write_model_summary(step=index)

    @abstractmethod
    def write_model_summary(self, step: int):
        raise NotImplementedError

    def samples_summary(self, data: Union[tf.Tensor, List[tf.Tensor]], step: int, suffix: str):
        if isinstance(data, list):
            if len(data) == 1:
                self.sample_summary(data=data[0], step=step, suffix=suffix)
            else:
                for i, sample in enumerate(data):
                    self.sample_summary(data=sample, step=step, suffix="{}_{}".format(suffix, i))
        else:
            self.sample_summary(data=data, step=step, suffix=suffix)

    @abstractmethod
    def sample_summary(self, data: tf.Tensor, step: int, suffix: str):
        raise NotImplementedError

    def extract_logged_modalities(self, modalities) -> List[tf.Tensor]:
        return to_list(extract_modality(modalities, self.logged_output_indices))

    @classmethod
    def from_model_and_subset(cls,
                              autoencoder: Callable,
                              subset: Union[SubsetLoader],
                              pattern: Pattern,
                              name: str,
                              is_train_callback: bool,
                              tensorboard: keras.callbacks.TensorBoard,
                              update_freq="epoch",
                              epoch_freq=1,
                              max_outputs=4,
                              inputs_are_outputs=True,
                              modality_index=None,
                              **kwargs
                              ) -> "ModalityCallback":
        batch = subset.get_batch(batch_size=4, pattern=pattern)

        if inputs_are_outputs:
            inputs = outputs = batch
        else:
            inputs, outputs = batch

        if modality_index is None:
            modality_index = 0

        one_shot_callback = cls(inputs=inputs, outputs=outputs, model=autoencoder,
                                tensorboard=tensorboard, is_train_callback=is_train_callback,
                                update_freq=update_freq, epoch_freq=epoch_freq,
                                logged_output_indices=modality_index,
                                name=name, max_outputs=max_outputs,
                                **kwargs)

        return one_shot_callback


# region Utility / wrappers
def extract_modality(data: Union[tf.Tensor, List[tf.Tensor]], indices: List[int]):
    if isinstance(data, list) or isinstance(data, tuple):
        return [data[i] for i in indices]
    else:
        return data
# endregion
