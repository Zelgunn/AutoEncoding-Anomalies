import tensorflow as tf
from typing import Union, List, Tuple, Dict

from data_processing import DataProcessor


class DataProcessorStack(DataProcessor):
    def __init__(self, data_processors: Dict[int, DataProcessor]):
        self.data_processors = data_processors

    def pre_process(self,
                    inputs: Union[tf.Tensor, List, Tuple]
                    ) -> Union[tf.Tensor, List, Tuple]:
        return self._process(inputs, pre_process=True)

    def post_process(self,
                     inputs: Union[tf.Tensor, List, Tuple]
                     ) -> Union[tf.Tensor, List, Tuple]:
        return self._process(inputs, pre_process=False)

    def _process(self,
                 inputs: Union[tf.Tensor, List, Tuple],
                 pre_process: bool,
                 ) -> Union[tf.Tensor, List, Tuple]:
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Inputs must either be a list or a tuple, not a {}".format(type(inputs)))

        outputs = []
        for i in range(len(inputs)):
            x = inputs[i]
            if i in self.data_processors:
                if pre_process:
                    x = self.data_processors[i].pre_process(x)
                else:
                    x = self.data_processors[i].post_process(x)
            outputs.append(x)

        if isinstance(inputs, tuple):
            outputs = tuple(outputs)

        return outputs
