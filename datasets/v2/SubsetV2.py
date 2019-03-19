from datasets import DatasetConfigV2


class SubsetV2(object):
    def __init__(self, config: DatasetConfigV2):
        self.config = config
