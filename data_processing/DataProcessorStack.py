from typing import List

from data_processing import DataProcessor


class DataProcessorStack(DataProcessor):
    def __init__(self, data_processors: List[DataProcessor]):
        self.data_processors = data_processors

    def pre_process(self, inputs):
        for data_processor in self.data_processors:
            inputs = data_processor.pre_process(inputs)
        return inputs

    def post_process(self, inputs):
        for data_processor in self.data_processors:
            inputs = data_processor.post_process(inputs)
        return inputs
