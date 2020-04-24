from typing import Dict

from data_processing import DataProcessor


class DataProcessorStack(DataProcessor):
    def __init__(self, data_processors: Dict[str, DataProcessor]):
        self.data_processors = data_processors
