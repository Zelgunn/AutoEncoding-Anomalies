import tensorflow as tf
from abc import abstractmethod, ABC
from typing import Union, List, Tuple


class DataProcessor(ABC):
    def pre_process(self,
                    inputs: Union[tf.Tensor, List, Tuple]
                    ) -> Union[tf.Tensor, List, Tuple]:
        return inputs

    def batch_process(self,
                      inputs: Union[tf.Tensor, List, Tuple]
                      ) -> Union[tf.Tensor, List, Tuple]:
        return inputs

    def post_process(self,
                     inputs: Union[tf.Tensor, List, Tuple]
                     ) -> Union[tf.Tensor, List, Tuple]:
        return inputs
