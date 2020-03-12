from tensorflow.python.keras.callbacks import TensorBoard
from typing import Union, List, Callable

from callbacks import ImageCallback
from callbacks.configs import ModalityCallbackConfig
from datasets import DatasetLoader
from modalities import Pattern


class ImageCallbackConfig(ModalityCallbackConfig):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 is_train_callback: bool,
                 name: str,
                 epoch_freq: int = 1,
                 video_sample_rate: int = 25,
                 inputs_are_outputs: bool = True,
                 modality_indices: Union[List[int], int] = None,
                 **kwargs,
                 ):
        super(ImageCallbackConfig, self).__init__(autoencoder=autoencoder,
                                                  pattern=pattern,
                                                  is_train_callback=is_train_callback,
                                                  name=name,
                                                  epoch_freq=epoch_freq,
                                                  inputs_are_outputs=inputs_are_outputs,
                                                  modality_indices=modality_indices,
                                                  **kwargs
                                                  )
        self.video_sample_rate = video_sample_rate

    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    seed=None,
                    ) -> ImageCallback:
        image_callbacks = ImageCallback.from_model_and_subset(autoencoder=self.autoencoder,
                                                              subset=self.get_subset(dataset_loader),
                                                              pattern=self.pattern,
                                                              name=self.name,
                                                              is_train_callback=self.is_train_callback,
                                                              tensorboard=tensorboard,
                                                              video_sample_rate=self.video_sample_rate,
                                                              epoch_freq=self.epoch_freq,
                                                              inputs_are_outputs=self.inputs_are_outputs,
                                                              modality_indices=self.modality_indices,
                                                              seed=seed,
                                                              **self.kwargs)
        return image_callbacks
