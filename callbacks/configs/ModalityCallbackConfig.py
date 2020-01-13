from tensorflow.python.keras.callbacks import TensorBoard
from abc import abstractmethod
from typing import List, Callable, Union

from datasets import DatasetLoader
from modalities import Pattern


class ModalityCallbackConfig(object):
    def __init__(self,
                 autoencoder: Callable,
                 pattern: Pattern,
                 is_train_callback: bool,
                 name: str,
                 epoch_freq: int = 1,
                 inputs_are_outputs: bool = True,
                 modality_indices: Union[List[int], int] = None,
                 **kwargs
                 ):
        self.autoencoder = autoencoder
        self.pattern = pattern
        self.name = name
        self.is_train_callback = is_train_callback
        self.epoch_freq = epoch_freq
        self.inputs_are_outputs = inputs_are_outputs
        self.modality_indices = modality_indices
        self.kwargs = kwargs

    def get_subset(self, dataset_loader: DatasetLoader):
        return dataset_loader.train_subset if self.is_train_callback else dataset_loader.test_subset

    @abstractmethod
    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    ) -> List:
        pass
