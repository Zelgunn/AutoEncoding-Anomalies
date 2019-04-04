class DatasetConfigV2(object):
    def __init__(self, input_sequence_length: int, output_sequence_length: int):
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        self.shard_size = None

    @property
    def sample_length(self):
        return self.output_sequence_length
