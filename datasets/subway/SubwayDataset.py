from typing import List

from datasets import PartiallyLoadableDataset
from data_preprocessors import DataPreprocessor


class SubwayDataset(PartiallyLoadableDataset):
    def __init__(self,
                 input_sequence_length: int or None,
                 output_sequence_length: int or None,
                 dataset_path: str,
                 epoch_length: int,
                 data_preprocessors: List[DataPreprocessor] = None,
                 batch_size=64,
                 shuffle_on_epoch_end=True,
                 **kwargs):
        super(SubwayDataset, self).__init__(input_sequence_length=input_sequence_length,
                                            output_sequence_length=output_sequence_length,
                                            dataset_path=dataset_path,
                                            data_preprocessors=data_preprocessors,
                                            batch_size=batch_size,
                                            epoch_length=epoch_length,
                                            shuffle_on_epoch_end=shuffle_on_epoch_end,
                                            **kwargs)

    def current_batch(self, batch_size: int = None, apply_preprocess_step=True):
        raise NotImplementedError


