from abc import abstractmethod, ABC


class DataProcessor(ABC):
    @abstractmethod
    def pre_process(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def post_process(self, inputs):
        raise NotImplementedError
