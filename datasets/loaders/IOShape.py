from typing import List, Union, Tuple


class IOShape(object):
    def __init__(self,
                 input_shape: Union[List[int], Tuple[int, ...]],
                 output_shape: Union[List[int], Tuple[int, ...]]):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def input_length(self) -> int:
        return self.input_shape[0]

    @property
    def output_length(self) -> int:
        return self.output_shape[0]

    @property
    def sample_length(self) -> int:
        return max(self.input_length, self.output_length)
