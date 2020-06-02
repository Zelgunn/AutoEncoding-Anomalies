import tensorflow as tf
from abc import abstractmethod, ABC
from typing import Union, List, Tuple


class DataProcessor(ABC):
    @abstractmethod
    def pre_process(self,
                    inputs: Union[tf.Tensor, List, Tuple]
                    ) -> Union[tf.Tensor, List, Tuple]:
        raise NotImplementedError

    @abstractmethod
    def post_process(self,
                     inputs: Union[tf.Tensor, List, Tuple]
                     ) -> Union[tf.Tensor, List, Tuple]:
        raise NotImplementedError
